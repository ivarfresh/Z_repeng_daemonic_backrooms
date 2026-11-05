"""
Backroom Engine
===============

Main conversation loop for infinite backroom conversations between
two LLM instances (one baseline, one with control vector).
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path to import repeng
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from repeng import ControlModel, ControlVector


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    control_vector_path: Optional[str] = None
    control_coeff: float = 0.0
    system_prompt: Optional[str] = None


def generate_with_logging(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    return_logits: bool = True,
    return_attention: bool = False
) -> tuple[str, dict]:
    """
    Generate text and return comprehensive metadata for logging.

    Args:
        model: Model instance (may be wrapped with ControlModel)
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        return_logits: Whether to return logits
        return_attention: Whether to return attention weights (warning: large!)

    Returns:
        Tuple of (response_text, metadata_dict)
        metadata contains: input_ids, output_ids, logits (optional), attentions (optional)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=return_logits,
            output_attentions=return_attention,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract generated tokens (excluding input)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0, input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metadata = {
        "input_ids": inputs["input_ids"].cpu(),
        "output_ids": generated_ids.cpu(),
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
    }

    if return_logits and outputs.scores:
        # outputs.scores is tuple of (vocab_size,) tensors, one per generated token
        # Stack into (num_tokens, vocab_size)
        metadata["logits"] = torch.stack(outputs.scores, dim=0).cpu()

    if return_attention and outputs.attentions:
        # outputs.attentions is complex nested structure
        # Store reference (will be processed in logger)
        metadata["attentions"] = outputs.attentions

    return response_text, metadata


class BackroomEngine:
    """
    Manages the infinite backroom conversation loop between two agents.
    """

    def __init__(
        self,
        model_name: str,
        agent_a_config: AgentConfig,
        agent_b_config: AgentConfig,
        device: str = "cuda",
        layer_ids: list = None
    ):
        """
        Initialize the backroom engine with two agent configurations.

        Args:
            model_name: HuggingFace model identifier
            agent_a_config: Configuration for agent A (typically baseline)
            agent_b_config: Configuration for agent B (typically with control vector)
            device: Device to use
            layer_ids: Layer IDs to wrap for control (default: -5 to -18)
        """
        self.model_name = model_name
        self.agent_a_config = agent_a_config
        self.agent_b_config = agent_b_config
        self.device = device

        # Default layer configuration
        self.layer_ids = layer_ids or list(range(-5, -18, -1))

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Wrap model for control
        print(f"Wrapping model with control layers: {self.layer_ids}")
        self.model = ControlModel(base_model, self.layer_ids)

        # Load control vectors
        self.agent_a_vector = None
        self.agent_b_vector = None

        if agent_a_config.control_vector_path:
            print(f"Loading Agent A vector: {agent_a_config.control_vector_path}")
            self.agent_a_vector = ControlVector.import_gguf(agent_a_config.control_vector_path)

        if agent_b_config.control_vector_path:
            print(f"Loading Agent B vector: {agent_b_config.control_vector_path}")
            self.agent_b_vector = ControlVector.import_gguf(agent_b_config.control_vector_path)

    def set_agent(self, agent_name: str):
        """
        Configure model for the specified agent.

        Args:
            agent_name: "A" or "B"
        """
        if agent_name == "A":
            config = self.agent_a_config
            vector = self.agent_a_vector
        elif agent_name == "B":
            config = self.agent_b_config
            vector = self.agent_b_vector
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        # Reset or set control vector
        if vector is None or config.control_coeff == 0.0:
            self.model.reset()
        else:
            self.model.set_control(vector, coeff=config.control_coeff)

    def generate_turn(
        self,
        prompt: str,
        agent_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        return_attention: bool = False
    ) -> tuple[str, dict]:
        """
        Generate a single conversation turn for the specified agent.

        Args:
            prompt: Input prompt
            agent_name: "A" or "B"
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (response_text, metadata)
        """
        # Configure for this agent
        self.set_agent(agent_name)

        # Generate
        response, metadata = generate_with_logging(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_logits=True,
            return_attention=return_attention
        )

        return response, metadata

    def run_session(
        self,
        logger,
        seed_prompt: str,
        num_turns: int,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        save_attention: bool = False
    ):
        """
        Run a complete backroom conversation session.

        Args:
            logger: ConversationLogger instance
            seed_prompt: Initial prompt to start conversation
            num_turns: Number of turns to generate
            max_new_tokens: Max tokens per turn
            temperature: Sampling temperature
            save_attention: Whether to save attention weights
        """
        # Set tokenizer in logger
        logger.set_tokenizer(self.tokenizer)

        # Initialize conversation
        current_prompt = f"[INST] {seed_prompt} [/INST]"

        print("\n" + "="*60)
        print(f"Starting session: {num_turns} turns")
        print("="*60)

        # Main loop
        for turn in range(num_turns):
            # Alternate agents
            if turn % 2 == 0:
                agent_name = "A"
                config = self.agent_a_config
                coeff = config.control_coeff
            else:
                agent_name = "B"
                config = self.agent_b_config
                coeff = config.control_coeff

            print(f"\nTurn {turn + 1}/{num_turns} | Agent {agent_name} (coeff={coeff:.2f})")

            # Generate
            response, metadata = self.generate_turn(
                prompt=current_prompt,
                agent_name=agent_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_attention=save_attention
            )

            # Log
            logger.log_turn(
                turn_id=turn,
                agent=agent_name,
                prompt=current_prompt,
                response=response,
                metadata=metadata,
                control_coeff=coeff
            )

            # Prepare next prompt
            current_prompt = f"[INST] {response} [/INST]"

            # Progress
            token_count = len(metadata.get("output_ids", []))
            print(f"  Generated {token_count} tokens")
            print(f"  Preview: {response[:100]}...")

        print("\n" + "="*60)
        print("Session complete!")
        print("="*60)
