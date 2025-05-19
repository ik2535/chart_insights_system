"""
LLM Wrapper for Chart Insights System.
Generic wrapper for different LLM providers.
"""

import os
import logging
import json
import requests
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class LLMWrapper:
    """
    Generic wrapper for different LLM providers.
    
    This class provides a unified interface for interacting with
    different LLM providers based on configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM wrapper.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.provider = config.get('llm', {}).get('provider', 'openai')
        self.model = config.get('llm', {}).get('model', 'gpt-4')
        self.api_key = config.get('llm', {}).get('api_key', os.environ.get('LLM_API_KEY', ''))
        self.temperature = config.get('llm', {}).get('temperature', 0.1)
        self.max_tokens = config.get('llm', {}).get('max_tokens', 2000)
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Generated response string
        """
        # Check if API key is available
        if not self.api_key and self.provider not in ['mock', 'local']:
            logger.warning(f"No API key provided for {self.provider}. Using mock LLM.")
            return self._generate_mock(prompt)
        
        # Call appropriate provider method
        if self.provider == 'openai':
            return self._generate_openai(prompt)
        elif self.provider == 'azure':
            return self._generate_azure(prompt)
        elif self.provider == 'anthropic':
            return self._generate_anthropic(prompt)
        elif self.provider == 'huggingface':
            return self._generate_huggingface(prompt)
        elif self.provider in ['mock', 'local']:
            return self._generate_mock(prompt)
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            return "Error: Unsupported LLM provider"
    
    def _generate_openai(self, prompt: str) -> str:
        """
        Generate a response using OpenAI's API.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Generated response string
        """
        try:
            import openai
            
            # Set API key
            openai.api_key = self.api_key
            
            # Create chat completion
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst that generates insights from chart data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and return content
            return response.choices[0].message.content
            
        except ImportError:
            logger.error("OpenAI package not installed. Run 'pip install openai'.")
            return "Error: OpenAI package not installed"
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            logger.error(f"Error generating response with {self.provider}: {e}")
            return self._generate_mock(prompt)
    
    def _generate_azure(self, prompt: str) -> str:
        """
        Generate a response using Azure OpenAI.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Generated response string
        """
        try:
            import openai
            
            # Azure-specific settings
            openai.api_type = "azure"
            openai.api_key = self.api_key
            openai.api_base = self.config.get('llm', {}).get('azure_endpoint', '')
            openai.api_version = self.config.get('llm', {}).get('azure_api_version', '2023-05-15')
            
            # Create chat completion
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst that generates insights from chart data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and return content
            return response.choices[0].message.content
            
        except ImportError:
            logger.error("OpenAI package not installed. Run 'pip install openai'.")
            return "Error: OpenAI package not installed"
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            return f"Error generating response: {e}"
    
    def _generate_anthropic(self, prompt: str) -> str:
        """
        Generate a response using Anthropic's API.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Generated response string
        """
        try:
            import anthropic
            
            # Create Anthropic client
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Create completion
            response = client.messages.create(
                model=self.model,
                system="You are an expert data analyst that generates insights from chart data.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and return content
            return response.content[0].text
            
        except ImportError:
            logger.error("Anthropic package not installed. Run 'pip install anthropic'.")
            return "Error: Anthropic package not installed"
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error generating response: {e}"
    
    def _generate_huggingface(self, prompt: str) -> str:
        """
        Generate a response using Hugging Face's API.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Generated response string
        """
        try:
            # HF inference API endpoint
            API_URL = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Prepare payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False
                }
            }
            
            # Make request
            response = requests.post(API_URL, headers=headers, json=payload)
            result = response.json()
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            else:
                return result.get('generated_text', str(result))
            
        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
            return f"Error generating response: {e}"
    
    def _generate_mock(self, prompt: str) -> str:
        """
        Generate a mock response for testing without LLM API.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Generated mock response string
        """
        logger.info("Using mock LLM for response generation")
        
        # Extract chart type and insight type from prompt
        chart_type = "unknown"
        insight_type = "unknown"
        
        if "CHART INFORMATION" in prompt:
            lines = prompt.split("\n")
            for line in lines:
                if "Chart Type:" in line:
                    chart_type = line.split(":", 1)[1].strip().lower()
        
        if "trend" in prompt.lower():
            insight_type = "trend"
        elif "comparison" in prompt.lower():
            insight_type = "comparison"
        elif "anomaly" in prompt.lower():
            insight_type = "anomaly"
        elif "correlation" in prompt.lower():
            insight_type = "correlation"
        
        # Generate mock insights based on chart and insight type
        insights = []
        
        if insight_type == "trend":
            insights = [
                "INSIGHT: The data shows an overall upward trend over the observed period.\n" +
                "CONFIDENCE: 0.85\n" +
                "EXPLANATION: The values consistently increase with minor fluctuations, indicating a positive trend.",
                
                "INSIGHT: There appears to be seasonal variation in the data.\n" +
                "CONFIDENCE: 0.72\n" +
                "EXPLANATION: The data shows recurring patterns with peaks and valleys at regular intervals."
            ]
        
        elif insight_type == "comparison":
            insights = [
                "INSIGHT: Category A has the highest value among all categories.\n" +
                "CONFIDENCE: 0.92\n" +
                "EXPLANATION: Category A's value is significantly higher than the average of other categories.",
                
                "INSIGHT: The bottom three categories combined account for less than 30% of the total.\n" +
                "CONFIDENCE: 0.78\n" +
                "EXPLANATION: The distribution is skewed with top categories representing a disproportionate share."
            ]
        
        elif insight_type == "anomaly":
            insights = [
                "INSIGHT: There is an unusual spike in July that deviates from the overall pattern.\n" +
                "CONFIDENCE: 0.88\n" +
                "EXPLANATION: The July value is over 2 standard deviations above the mean, indicating an anomaly.",
                
                "INSIGHT: The December value is abnormally low compared to previous months.\n" +
                "CONFIDENCE: 0.76\n" +
                "EXPLANATION: This decrease breaks the established pattern and may indicate an unusual event."
            ]
        
        elif insight_type == "correlation":
            insights = [
                "INSIGHT: There is a strong positive correlation between variables X and Y.\n" +
                "CONFIDENCE: 0.89\n" +
                "EXPLANATION: As X increases, Y consistently increases as well, with a correlation coefficient of approximately 0.9.",
                
                "INSIGHT: The correlation weakens at higher values.\n" +
                "CONFIDENCE: 0.71\n" +
                "EXPLANATION: Above a certain threshold, the relationship becomes less predictable."
            ]
        
        else:
            insights = [
                "INSIGHT: The data reveals interesting patterns that warrant further investigation.\n" +
                "CONFIDENCE: 0.75\n" +
                "EXPLANATION: While specific conclusions are difficult to draw without more context, there are clear structures in the data."
            ]
        
        # Adapt insights based on chart type
        if chart_type == "line":
            insights[0] = ("INSIGHT: The line chart shows a clear upward trend over time.\n"
                          "CONFIDENCE: 0.87\n"
                          "EXPLANATION: The line consistently moves upward with some minor fluctuations.")
        
        elif chart_type == "bar":
            insights[0] = ("INSIGHT: The 'Food' category has the highest value among all categories.\n"
                          "CONFIDENCE: 0.91\n"
                          "EXPLANATION: The bar representing Food is significantly taller than others.")
        
        elif chart_type == "pie":
            insights[0] = ("INSIGHT: The top three segments account for over 70% of the total.\n"
                          "CONFIDENCE: 0.84\n"
                          "EXPLANATION: The pie chart shows a clear dominance of a few key segments.")
        
        elif chart_type == "scatter":
            insights[0] = ("INSIGHT: There is a strong positive correlation between the variables.\n"
                          "CONFIDENCE: 0.88\n"
                          "EXPLANATION: The scatter plot points form a clear upward pattern.")
        
        # Join insights and return
        return "\n\n".join(insights)
