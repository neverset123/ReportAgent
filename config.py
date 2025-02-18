from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class Section:
    name: str
    keywords: List[str]
    description: str
    label: str
    threshold: float

@dataclass
class SectionConfig:
    sections: List[Section] = field(default_factory=list)

    def get_config_names(self) -> List[str]:
        return [section.name for section in self.sections]
    
    def get_config(self, name: str) -> Optional[dict]:
        for section in self.sections:
            if section.name == name:
                return {"keywords": section.keywords, "description": section.description, "label": section.label, "threshold": section.threshold}
        return None
    
    def get_all_configs(self) -> List[Dict[str, str]]:
        return [{"name": section.name, "description": section.description} for section in self.sections]

config = SectionConfig(sections=[
    Section(
        name="RAG",
        keywords=["rag", "recommendation"],
        description="RAG (Retrieval-Augmented Generation) is a machine learning approach that combines information retrieval from external sources with generative models to improve the accuracy and relevance of responses.",
        label="rag",
        threshold=0.5
    ),
    Section(
        name="LLM",
        keywords=["llm", "deepseek"],
        description="A Large Language Model (LLM) is an AI model trained on vast amounts of text data to understand, generate, and manipulate human language with high proficiency.",
        label="llm",
        threshold=0.6
    ),
    Section(
        name="Autonomous Driving",
        keywords=["autonomous driving", "perception"],
        description="Scene understanding in autonomous driving involves interpreting and analyzing the environment around a vehicle using sensors and AI to make real-time decisions for safe navigation.",
        label="ad",
        threshold=0.5
    ),
    Section(
        name="Data Mining",
        keywords=["data mining"],
        description="Data mining in automotive refers to the process of analyzing large datasets from vehicles and sensors to uncover patterns, trends, and insights that can improve performance, safety, and customer experience.",
        label="mining",
        threshold=0.6
    ),
    Section(
        name="CLIP",
        keywords=["clip"],
        description="CLIP (Contrastive Language-Image Pretraining) is a model that enables multimodal search by learning to connect images and text, allowing for more accurate and flexible search across visual and textual data.",
        label="clip",
        threshold=0.6
    )
])
