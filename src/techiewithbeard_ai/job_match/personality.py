"""
Resume Personality Configuration

Defines different critique and tailoring styles for different HR markets.
- American: Results-driven, metrics-focused, action-oriented
- European: Detail-oriented, quality-focused, collaborative
- Tech: Innovation-focused, technical depth, growth-oriented
"""

from enum import Enum
from pydantic import BaseModel, Field


class ResumePersonality(str, Enum):
    """Resume critique and tailoring personality styles"""
    AMERICAN = "american"
    EUROPEAN = "european"
    TECH_STARTUP = "tech_startup"
    ENTERPRISE = "enterprise"


class PersonalityConfig(BaseModel):
    """Configuration for resume critique and tailoring personality"""
    
    personality: ResumePersonality = Field(
        default=ResumePersonality.AMERICAN,
        description="Resume personality style"
    )
    
    # System prompt customization
    critique_style: str = ""  # Will be populated based on personality
    skill_matching_style: str = ""  # Will be populated based on personality
    bullet_enhancement_style: str = ""  # Will be populated based on personality


def get_personality_config(personality: ResumePersonality) -> PersonalityConfig:
    """Get personality configuration based on selected style"""
    
    configs = {
        ResumePersonality.AMERICAN: PersonalityConfig(
            personality=ResumePersonality.AMERICAN,
            critique_style="""
AMERICAN HR TECH FIRM PERSPECTIVE:
- Focus on measurable results and impact (numbers, percentages, growth)
- Emphasize speed, efficiency, and business value
- Use action verbs: "led", "drove", "achieved", "delivered"
- Highlight innovation and competitive advantage
- Value quick wins and MVP mentality
- Prefer concise, punchy language
- Examples: "Increased performance by 40%", "Launched MVP in 2 weeks"
- Critique should identify gaps quickly and suggest immediate improvements
""",
            skill_matching_style="""
Prioritize skills that:
1. Directly match job requirements (100% match preferred)
2. Show proven business impact
3. Demonstrate scalability and growth mindset
4. Highlight unique/valuable combinations
Reorder experience bullets to emphasize most relevant achievements first.
""",
            bullet_enhancement_style="""
Enhance bullets by:
- Adding metrics/numbers where possible
- Emphasizing business outcomes, not just tasks
- Using strong action verbs
- Highlighting individual contribution
- Keep sentences under 20 words when possible
Example: "Designed UI component library" → "Built reusable UI component library used by 15+ teams, reducing dev time by 30%"
""",
        ),
        
        ResumePersonality.EUROPEAN: PersonalityConfig(
            personality=ResumePersonality.EUROPEAN,
            critique_style="""
EUROPEAN HR TECH FIRM PERSPECTIVE:
- Emphasis on quality, thoroughness, and professional development
- Value collaboration, team contribution, and knowledge sharing
- Use measured language: "contributed to", "supported", "facilitated"
- Focus on long-term impact and sustainable growth
- Appreciate certifications, training, and continuous learning
- Prefer complete context and detailed explanations
- Examples: "Collaborated with team to improve processes", "Developed robust documentation"
- Critique should provide constructive feedback with context
- Value work-life balance indicators and team culture contribution
""",
            skill_matching_style="""
Prioritize skills that:
1. Show depth and expertise development over time
2. Demonstrate team collaboration and knowledge transfer
3. Include certifications and professional development
4. Show adaptability and learning across multiple domains
Ensure experience bullets reflect quality and professional growth.
""",
            bullet_enhancement_style="""
Enhance bullets by:
- Adding context and impact on team/organization
- Including professional development aspects
- Emphasizing quality and robustness
- Highlighting collaboration and mentoring
- Provide sufficient detail for understanding
Example: "Designed UI component library" → "Designed and documented comprehensive UI component library following accessibility standards, enabling team collaboration and reducing design-implementation gaps"
""",
        ),
        
        ResumePersonality.TECH_STARTUP: PersonalityConfig(
            personality=ResumePersonality.TECH_STARTUP,
            critique_style="""
TECH STARTUP PERSPECTIVE:
- Focus on innovation, learning velocity, and technical depth
- Emphasize problem-solving and ownership mindset
- Value experimentation and failure as learning
- Use contemporary tech language: "shipped", "iterated", "scaled", "optimized"
- Highlight full-stack capabilities and cross-functional impact
- Appreciate architectural decisions and technical tradeoffs
- Examples: "Architected microservices", "Shipped feature impacting 1M+ users"
- Critique should identify growth opportunities and skill gaps
- Value side projects, open source, and continuous learning
""",
            skill_matching_style="""
Prioritize skills that:
1. Show technical depth and breadth
2. Demonstrate shipping capability and iteration
3. Include emerging/cutting-edge technologies
4. Show autonomous problem-solving and ownership
Reorder to highlight most technically impressive and relevant work first.
""",
            bullet_enhancement_style="""
Enhance bullets by:
- Adding technical depth and architectural decisions
- Emphasizing scale and performance improvements
- Highlighting learning and iteration
- Including technology stack and tools used
- Focus on shipped impact, not just effort
Example: "Designed UI component library" → "Architected and shipped monorepo-based UI component library (React + TypeScript), enabling 15+ teams to ship features 2x faster with 99.9% uptime"
""",
        ),
        
        ResumePersonality.ENTERPRISE: PersonalityConfig(
            personality=ResumePersonality.ENTERPRISE,
            critique_style="""
ENTERPRISE ORGANIZATION PERSPECTIVE:
- Emphasis on stability, governance, and risk management
- Value standardization, process improvement, and compliance
- Use formal language: "implemented", "managed", "administered", "coordinated"
- Focus on organizational impact and cross-department alignment
- Appreciate documentation, training, and knowledge management
- Value experience with large-scale systems and legacy modernization
- Examples: "Managed infrastructure for 10K+ users", "Implemented company-wide standards"
- Critique should focus on organizational fit and risk reduction
- Value security, reliability, and regulatory compliance indicators
""",
            skill_matching_style="""
Prioritize skills that:
1. Demonstrate experience at scale (large organizations)
2. Show governance and compliance awareness
3. Include enterprise technologies and platforms
4. Show ability to work in structured environments
Emphasize stability, reliability, and organizational contributions.
""",
            bullet_enhancement_style="""
Enhance bullets by:
- Adding scale and scope of impact (users, systems, budget)
- Emphasizing governance, compliance, and risk management
- Highlighting organizational initiatives and standards
- Including certification or compliance aspects
- Focus on reliability, maintainability, and long-term value
Example: "Designed UI component library" → "Established standardized UI component library with comprehensive documentation and training program, ensuring compliance across 50+ internal applications and reducing design-code discrepancies by 80%"
""",
        ),
    }
    
    return configs.get(personality, configs[ResumePersonality.AMERICAN])


def get_personality_description(personality: ResumePersonality) -> str:
    """Get human-readable description of personality"""
    descriptions = {
        ResumePersonality.AMERICAN: "🇺🇸 American Tech - Results-driven, metrics-focused, action-oriented",
        ResumePersonality.EUROPEAN: "🇪🇺 European - Quality-focused, collaborative, sustainable growth",
        ResumePersonality.TECH_STARTUP: "🚀 Tech Startup - Innovation, technical depth, ownership mindset",
        ResumePersonality.ENTERPRISE: "🏢 Enterprise - Stability, governance, large-scale impact",
    }
    return descriptions.get(personality, "Unknown")
