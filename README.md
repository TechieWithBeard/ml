Techie With Beard AI Lab 🧠
A hands-on AI/ML experimentation workspace focused on building practical LLM-powered applications using Python, LangChain, LangGraph, RAG, local models, vector databases, Pydantic, and Streamlit.
The goal is to explore how AI concepts can be turned into practical, production-oriented applications.
🚀 Current Projects
1. Resume Analyzer & RAG Application
A Streamlit-based Resume Analyzer that allows users to:
📄 Upload a PDF resume
🔍 Extract and chunk document content
🧠 Generate embeddings using Ollama
🗄️ Store embeddings in Chroma Cloud
🔎 Retrieve relevant resume sections using semantic similarity search
🤖 Ask questions about the uploaded resume
📦 Generate structured responses using Pydantic models
🔗 Analyze external profile links as part of the next stage
2. AI Job Match Analyzer
A LangGraph-powered workflow that analyzes a candidate's resume against a job description.
The workflow currently:
📄 Parses the candidate's resume
💼 Extracts technical requirements from the job description
🔎 Matches candidate skills against required skills
⚠️ Identifies missing skills
🔄 Analyzes skill transferability
📊 Calculates an overall candidate score
📝 Generates a candidate critique
🕸️ Visualizes the workflow using Mermaid
The application currently runs locally using Ollama + Gemma, allowing the entire analysis pipeline to run without relying on a hosted LLM API.
🧠 What I'm Learning
This project is also a learning lab for exploring:
LangChain
LangGraph
RAG architectures
Prompt engineering
Structured LLM output
Pydantic validation
Local LLMs with Ollama
Embedding models
Vector databases
Semantic search
Agentic workflows
Streamlit
AI application architecture
LLM performance optimization
🏗️ Current RAG Architecture
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    │                     │
                    │  PDF Upload         │
                    │  Resume Q&A         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    PDF Loader       │
                    │                     │
                    │ PyPDFLoader         │
                    │ Text Splitting      │
                    │ File Hashing        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Ollama Embeddings   │
                    │                     │
                    │ embeddinggemma      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Chroma Cloud      │
                    │                     │
                    │ Vector Storage      │
                    │ Semantic Search     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     RAG Chain       │
                    │                     │
                    │ Retrieve Context    │
                    │ Build Prompt        │
                    │ Generate Answer     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Ollama LLM      │
                    │                     │
                    │       Gemma         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Structured Output  │
                    │                     │
                    │  Pydantic Models    │
                    │  ResumeAnalyzer     │
                    └─────────────────────┘
​
🔄 Job Match Architecture
                    ┌─────────────────────┐
                    │     Resume PDF      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Parse Resume     │
                    │                     │
                    │ Candidate Name      │
                    │ Skills              │
                    │ Experience          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Parse Requirements  │
                    │                     │
                    │ Required Skills     │
                    │ Experience          │
                    │ Responsibilities    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Match Skills     │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
              Missing Skills?            │
                     │                   │
                    Yes                  No
                     │                   │
                     ▼                   │
          ┌─────────────────────┐        │
          │ Transferability     │        │
          │ Analysis            │        │
          └──────────┬──────────┘        │
                     │                   │
                     └─────────┬─────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Calculate Score     │
                    │                     │
                    │ Skill Score         │
                    │ Experience Score    │
                    │ Responsibility Score│
                    │ Overall Score       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Generate Critique    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Final Report     │
                    └─────────────────────┘
​
### Sample response
```JSON
{
  "candidate_name": "VISHNU THANKAPPAN",
  "candidate_skills": [
    "Angular",
    "Nx",
    "TypeScript",
    "JavaScript",
    "HTML",
    "CSS",
    "SCSS",
    "Design Systems",
    "Modular UI Architecture",
    "NgRx",
    "Jasmine",
    "Karma",
    "Cypress",
    "Playwright",
    "Git",
    "CI/CD",
    "Azure",
    "Storybook",
    "REST APIs",
    "Webpack",
    "Ionic",
    "Xamarin",
    "Agile/Scrum",
    "Distributed Teams",
    "Power Apps",
    "Power Platform"
  ],
  "candidate_experience": [
    "Experience(title='Senior Frontend Engineer', company='Parnasoft Technologies — Client: AVEVA', duration=None, responsibilities=[])",
    "Experience(title='Expert Frontend Engineer', company='Maistering B.V.', duration=None, responsibilities=[])",
    "Experience(title='Frontend Specialist', company='ACI Logistics', duration=None, responsibilities=[])"
  ],
  "required_skills": [
    "Javascript",
    "HTML/CSS",
    "Web Development",
    "Advanced Programming Skills",
    "Modern frameworks (Angular, Ember, React)",
    "Debugging tools (Chrome Dev Tools)",
    "Python",
    "Java",
    "C/C++",
    "C#",
    "Objective-C",
    "Ruby",
    "Expert-level knowledge of Ember",
    "Knowledge of current trends and best practices in front-end architecture (performance, accessibility, security, usability)",
    "Exposure to Android or IOS development"
  ],
  "required_experience": [
    "BA/BS in Computer Science or related technical field or equivalent practical experience",
    "12+ years of industry experience in front-end software design, development, and algorithm related solutions",
    "12+ years programming experience in languages such as Javascript, HTML, CSS, Python, Java, C/C++, C#, Objective-C, Ruby, etc.",
    "Experience writing clean JavaScript including experience with modern frameworks (Angular/Ember/React) and debugging tools (Chrome Dev Tools, etc.)",
    "Prior experience building public API's with Java (Preferred)",
    "Experience building large-scale consumer facing products (Preferred)"
  ],
  "responsibilities": [
    "Own the front-end development for one or more products.",
    "Collaborate with visual/interaction designers, other engineers, and product managers to launch new products, iterate on existing features, and build a world-class user experience.",
    "Implement cutting-edge technologies and write state-of-the-art code.",
    "Ensure the site is delightful, secure, performant, and accessible to all members.",
    "Meet with colleagues including product managers, designers, and other engineers assigned to projects.",
    "Provide architectural guidance and mentorship to up-level the engineering organization.",
    "Develop best practices and define best strategies for craftsmanship.",
    "Act as a role model and professional coach for engineers.",
    "Identify, leverage, and successfully evangelize opportunities and collaborate with cross functional teams to design and build scalable platforms/products/services/tools and to improve engineering productivity in the organization.",
    "Work with peers across teams to support and leverage a shared technical stack.",
    "Resolve conflicts between teams within the organization to get alignment and build team culture.",
    "Review others' work and share knowledge."
  ],
  "matching_skills": [
    "Javascript",
    "HTML/CSS",
    "Web Development",
    "Modern frameworks (Angular, Ember, React)",
    "C#",
    "Exposure to Android or IOS development"
  ],
  "missing_skills": [
    "Advanced Programming Skills",
    "Debugging tools (Chrome Dev Tools)",
    "Python",
    "Java",
    "C/C++",
    "Objective-C",
    "Ruby",
    "Expert-level knowledge of Ember",
    "Knowledge of current trends and best practices in front-end architecture (performance, accessibility, security, usability)"
  ],
  "skill_matches": [
    "SkillMatch(requirement='Javascript', matched=True, evidence=\"The skill 'JavaScript' is explicitly listed in the candidate skills.\", confidence=1.0)",
    "SkillMatch(requirement='HTML/CSS', matched=True, evidence=\"The candidate list explicitly includes both 'HTML' and 'CSS'.\", confidence=1.0)",
    "SkillMatch(requirement='Web Development', matched=True, evidence='The candidate lists core web technologies (HTML, CSS, JavaScript) alongside major frameworks and tools specific to modern web development (Angular, TypeScript, Webpack, Storybook, Design Systems).', confidence=1.0)",
    "SkillMatch(requirement='Advanced Programming Skills', matched=False, evidence=None, confidence=0.95)",
    "SkillMatch(requirement='Modern frameworks (Angular, Ember, React)', matched=True, evidence=\"The candidate explicitly lists 'Angular', which is one of the required modern frameworks.\", confidence=1.0)",
    "SkillMatch(requirement='Debugging tools (Chrome Dev Tools)', matched=False, evidence=None, confidence=0.95)",
    "SkillMatch(requirement='Python', matched=False, evidence=None, confidence=1.0)",
    "SkillMatch(requirement='Java', matched=False, evidence=None, confidence=1.0)",
    "SkillMatch(requirement='C/C++', matched=False, evidence=None, confidence=1.0)",
    "SkillMatch(requirement='C#', matched=True, evidence='Xamarin (A framework that fundamentally requires proficiency in C#)', confidence=0.95)",
    "SkillMatch(requirement='Objective-C', matched=False, evidence=None, confidence=1.0)",
    "SkillMatch(requirement='Ruby', matched=False, evidence=None, confidence=1.0)",
    "SkillMatch(requirement='Expert-level knowledge of Ember', matched=False, evidence=None, confidence=1.0)",
    "SkillMatch(requirement='Knowledge of current trends and best practices in front-end architecture (performance, accessibility, security, usability)', matched=False, evidence=None, confidence=0.95)",
    "SkillMatch(requirement='Exposure to Android or IOS development', matched=True, evidence='Xamarin (A dedicated framework for building native cross-platform applications targeting both iOS and Android)', confidence=1.0)"
  ],
  "transferability": [
    "Transferability(missing_skill='Advanced Programming Skills', related_skills=[], transferability_score=80.0, learning_difficulty='Moderate to High', reasoning=\"The missing skill 'Advanced Programming Skills' is extremely broad and generally refers to deep knowledge of computer science fundamentals (e.g., complex data structures, algorithms, time/space complexity analysis) that go beyond typical framework usage. However, the candidate's profile strongly suggests a high level of conceptual understanding necessary for advanced programming.\\n\\n**Related Skills:** The most relevant skills are **TypeScript**, **JavaScript**, and especially **Modular UI Architecture** and **NgRx**. Implementing complex state management (NgRx) or designing highly modular systems requires more than just knowing syntax; it demands an understanding of data flow, computational efficiency, and architectural patterns. Furthermore, experience with build tools like **Webpack** implies a conceptual understanding of how code is compiled, optimized, and executed—a core advanced programming concept.\\n\\n**Analysis:** The candidate's seniority (Senior/Expert titles) combined with their mastery of complex frontend concepts (like state management and modularity) indicates they have already operated at an advanced level within the web stack. They are not merely a coder; they are an architect who understands *why* certain patterns are used. While we cannot confirm proficiency in general computer science algorithms or low-level languages, their existing foundation is robust enough that transitioning into structured learning (e.g., algorithmic problem sets) would be highly effective.\")",
    "Transferability(missing_skill='Debugging tools (Chrome Dev Tools)', related_skills=['TypeScript', 'JavaScript', 'Angular', 'NgRx', 'Cypress', 'Playwright', 'Webpack', 'Senior Frontend Engineer/Expert Frontend Engineer titles'], transferability_score=95.0, learning_difficulty='Low', reasoning=\"The ability to use advanced debugging tools is not a standalone skill but a fundamental operational requirement for any professional working with modern web frameworks and complex testing suites. The candidate's listed skills and experience strongly imply mastery of these tools.\")",
    "Transferability(missing_skill='Python', related_skills=[], transferability_score=75.0, learning_difficulty='Low to Moderate', reasoning=\"The candidate is an experienced 'Expert Frontend Engineer' who has demonstrated mastery in a complex, high-level programming language (TypeScript/JavaScript). This indicates strong foundational knowledge of core computer science concepts: control flow, data structures, object-oriented principles, and algorithmic thinking. These conceptual skills are highly transferable to any other general-purpose language like Python.\\n\\nThe primary hurdle is not the concept, but the syntax and ecosystem. Python uses a distinct syntax (e.g., indentation for blocks, different function definitions) compared to JavaScript/TypeScript. However, because the candidate has proven experience in complex software development cycles (CI/CD, Angular architecture), they possess the discipline and ability to rapidly learn new syntaxes and paradigms.\\n\\nWhile their current focus is client-side web development, Python's versatility means that learning it for backend scripting or data processing would be a natural extension of their engineering mindset. The transferability score reflects strong conceptual aptitude but acknowledges the necessary effort required to master an entirely new language syntax.\")",
    "Transferability(missing_skill='Java', related_skills=[], transferability_score=65.0, learning_difficulty='Medium to High', reasoning='The candidate possesses strong foundational programming skills demonstrated by their expertise in TypeScript and Angular. Both languages are high-level, object-oriented paradigms, meaning the candidate has a solid grasp of core concepts like classes, inheritance, data structures, control flow, and modular architecture (OOP principles). This conceptual understanding is highly transferable.\\n\\nHowever, Java operates on a different syntax, memory model, and ecosystem (JVM) compared to JavaScript/TypeScript. While the *logic* required for software development is portable, mastering Java requires learning its specific grammar, standard libraries, and typical backend frameworks (e.g., Spring Boot). The transition from a frontend-focused language like TypeScript to an enterprise backend language like Java represents a significant paradigm shift in tooling and architecture.\\n\\n**Conclusion:** The candidate has the intellectual capacity and foundational programming maturity to learn Java. They are not starting from scratch. However, they will need dedicated time to overcome the syntax differences and adapt their architectural mindset from client-side rendering/state management (NgRx) to server-side business logic.')",
    "Transferability(missing_skill='C/C++', related_skills=[], transferability_score=25.0, learning_difficulty='High', reasoning='The candidate possesses a strong background in modern, high-level web development (TypeScript, Angular, JavaScript). These languages operate within managed environments with automatic memory management (garbage collection), abstracting away low-level details like pointers and manual resource allocation. C/C++, conversely, are compiled, low-level languages that require explicit, manual memory management (pointers, stack vs. heap, RAII). While the candidate demonstrates exceptional architectural understanding and problem-solving ability (evidenced by mastering complex frameworks like Angular/Nx and tools like Webpack), the core paradigms of C++ are fundamentally different from those they currently use. The transition requires learning entirely new concepts related to system architecture and memory handling that are orthogonal to their existing expertise. This is a significant paradigm shift, making it challenging but not impossible for an experienced engineer.')",
    "Transferability(missing_skill='Objective-C', related_skills=[], transferability_score=60.0, learning_difficulty='Medium to High', reasoning=\"The candidate's background is heavily rooted in modern, object-oriented web development (Angular/TypeScript). Objective-C is a mature, C-based language primarily used for Apple platform development. The primary transferable skill is the strong grasp of **Object-Oriented Programming (OOP) principles** and structured coding practices demonstrated by their use of Angular, TypeScript, and NgRx.\\n\\nHowever, the transition involves significant conceptual shifts: Objective-C syntax is fundamentally different from JavaScript/TypeScript, requiring familiarity with C-style memory management and pointers. While the candidate has experience with cross-platform frameworks like **Xamarin** (which bridges web concepts to native mobile code), this suggests an openness to non-web development paradigms.\\n\\nThe core challenge is not the concept of OOP, but mastering a specific, older syntax and ecosystem that differs greatly from their current JavaScript/TypeScript environment. This requires dedicated study outside of standard web development practices.\")",
    "Transferability(missing_skill='Ruby', related_skills=[], transferability_score=75.0, learning_difficulty='Moderate-High', reasoning=\"The candidate's profile demonstrates mastery of complex programming paradigms (TypeScript/Angular) and the ability to integrate with backend services via REST APIs. This proves a high aptitude for learning new languages and understanding core computer science concepts (data structures, control flow, OOP principles). Ruby is a general-purpose language that shares fundamental programming logic with JavaScript/TypeScript. The primary challenge is not the concept of programming itself, but the shift in syntax, runtime environment, and ecosystem (from JS/Node.js to Ruby/Rails). Given their 'Expert' level experience across multiple complex stacks, they possess the foundational knowledge required to quickly grasp a new language structure, even if it requires dedicated effort to master the specific idioms of Ruby.\")",
    "Transferability(missing_skill='Expert-level knowledge of Ember', related_skills=[], transferability_score=75.0, learning_difficulty='Moderate', reasoning=\"The candidate has demonstrated deep expertise in mastering large, opinionated, component-based frameworks (Angular). This proves a high capacity for learning complex architectural patterns and specific framework APIs. Ember is fundamentally another JavaScript framework built on similar principles (MVC/MVVM, components, lifecycle hooks). The conceptual transferability is very high.\\n\\nHowever, 'Expert-level' knowledge implies deep institutional understanding of Ember's unique conventions, internal workings, and best practices—knowledge that cannot be assumed simply because the candidate knows Angular. While they can quickly learn the syntax and basic usage, achieving true expert status in a new framework requires dedicated time to absorb its specific paradigms (e.g., data binding, routing structure) which differ significantly from those used in Angular/TypeScript.\\n\\nThe existing skills confirm that the candidate is an advanced practitioner capable of mastering complex technologies, making the learning curve manageable but requiring significant effort to reach expert status.\")",
    "Transferability(missing_skill='Knowledge of current trends and best practices in front-end architecture (performance, accessibility, security, usability)', related_skills=[], transferability_score=80.0, learning_difficulty='Low to Moderate', reasoning=\"This missing skill is a meta-skill—it represents deep industry knowledge rather than a specific technology. The candidate's existing profile demonstrates strong practical experience in implementing modern architectural patterns and development workflows, which are the primary components of this missing skill.\\n\\n**Strengths:** The presence of 'Modular UI Architecture', 'Design Systems', 'Storybook', and 'Nx' indicates that the candidate has already adopted best practices for componentization, reusability, and maintainability. Experience with 'CI/CD', 'Webpack', and 'Azure' suggests familiarity with modern deployment pipelines and build optimization (performance). Furthermore, roles like 'Senior Frontend Engineer' imply exposure to complex, real-world problems requiring architectural decision-making.\\n\\n**Gaps:** While the candidate has implemented these practices, the profile does not explicitly confirm deep theoretical knowledge in specific areas such as advanced WCAG compliance details (Accessibility), or specialized security protocols beyond basic API integration (Security). However, given their seniority and breadth of skills across multiple frameworks (Angular, Ionic, Xamarin) and platforms (Power Platform), they have the conceptual foundation to quickly absorb these best practices through focused study and mentorship. The learning curve is primarily about deepening theoretical knowledge rather than mastering a new technical stack.\")"
  ],
  "experience_score": 100,
  "responsibility_score": 100,
  "overall_score": 70,
  "critique": {
    "strengths": [],
    "weaknesses": [],
    "missing_skills": [],
    "learning_potential": "High",
    "recommendations": [
      "**Proceed to the interview stage.** The candidate possesses deep expertise in modern frontend architecture and has demonstrated mastery of complex, enterprise-level tools (Angular, Nx, Storybook). Their skill set is highly relevant to owning front-end development for large products.",
      "**Interviewer Focus Areas:**",
      "1. **Architectural Depth:** Test their ability to provide architectural guidance and define best practices. Ask scenario-based questions regarding state management optimization (NgRx) or component lifecycle issues in a large, modular system.",
      "2. **Cross-Functional Leadership:** Since the role requires mentoring and resolving team conflicts, assess their communication skills, experience collaborating with Product Managers/Designers, and ability to evangelize technical best practices across teams.",
      "3. **System Design:** Present a complex feature requirement (e.g., building a real-time dashboard) and ask them to outline the entire front-end architecture, including data flow, component breakdown, and testing strategy (Cypress/Playwright).",
      "**Technical Areas to Test:** TypeScript proficiency, Modular UI Architecture implementation, State Management patterns (NgRx), and CI/CD pipeline integration."
    ],
    "overall_assessment": "Strong Candidate - Proceed with Interviewing."
  }
}

```


⚡ Performance Work in Progress
The Job Match Analyzer currently runs completely locally using Ollama + Gemma.
The current end-to-end workflow takes around 5 minutes on my local setup.
The next focus is improving performance and identifying bottlenecks:
⏱️ Identify the slowest nodes
🔍 Profile individual LLM calls
📉 Reduce unnecessary model invocations
🔄 Explore parallel LangGraph execution
🧩 Optimize prompts and context size
🧠 Evaluate model configuration and token limits
⚡ Improve structured-output performance
📊 Measure latency across the complete workflow
The goal is to understand where the bottlenecks are and progressively improve the architecture rather than simply switching to a larger or faster hosted model.
🛠️ Tech Stack
Python
LangChain
LangGraph
Ollama
Gemma
Pydantic
Chroma
Streamlit
PyPDF
PDM
📁 Project Structure
ml/
├── data/
├── notebooks/
├── streamlit/
│   ├── app.py
│   └── pages/
│       └── job_match.py
├── src/
│   └── techiewithbeard_ai/
│       ├── agents/
│       ├── chains/
│       ├── job_match/
│       │   ├── graph.py
│       │   ├── nodes.py
│       │   ├── schemas.py
│       │   └── state.py
│       ├── loaders/
│       └── schema/
├── tests/
├── pyproject.toml
└── README.md
​
🚧 What's Next?
Improve Job Match Analyzer performance
Add better experience and responsibility matching
Improve transferability analysis
Add explainable scoring
Experiment with parallel LangGraph nodes
Add evaluation and benchmarking
Explore different local models
Experiment with hosted LLMs
Expand RAG capabilities
Integrate the projects into my developer portfolio
🔗 Project
GitHub: [Add your GitHub repository link here]
This repository is an ongoing AI/ML learning and experimentation lab.
The focus is not just on building AI features, but understanding the architecture, trade-offs, performance, and engineering challenges behind them.