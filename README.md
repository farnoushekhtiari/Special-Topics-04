# Special Topics in Computer Engineering

A comprehensive course covering modern full-stack development, from version control and containerization to web crawling, database optimization, and AI-assisted frontend development. This course equips computer engineering students with practical skills for building end-to-end web applications.

## Course Overview

This course follows a progressive learning path, starting with foundational tools and concepts, then building toward advanced topics and culminating in a complete full-stack project. Students will learn to design, implement, and deploy modern web applications using industry-standard technologies.

## Learning Objectives

By the end of this course, students will be able to:
- Manage software projects using Git and GitHub workflows
- Containerize applications with Docker
- Build robust REST APIs with FastAPI
- Implement efficient data pipelines with web crawling and database storage
- Optimize database performance and implement authentication
- Generate frontend interfaces using AI tools
- Deploy and manage complete full-stack applications

## Course Structure

### Section 1: Essential Tools & Workflows
**Subject 1: Git Basics & Project Workflows**
- Version control fundamentals
- Branching strategies and merge conflict resolution
- Team collaboration workflows
- Pull request management

**Subject 2: Virtual Environments, Dependency Management & Uvicorn**
- Python virtual environments
- Dependency management with requirements.txt/pyproject.toml
- ASGI application deployment with Uvicorn
- Reproducible development environments

**Subject 3: Project Management with Issues/Boards and Basic CI**
- GitHub Issues and project boards
- Milestone planning and issue tracking
- Continuous Integration with GitHub Actions
- Automated testing and linting

**Subject 4: Dockerizing Projects & Docker Fundamentals**
- Container fundamentals and Docker images
- Dockerfile creation and multi-stage builds
- Docker Compose for multi-service applications
- Container networking and volumes

### Section 2: Backend Development with FastAPI
**Subject 5: FastAPI Fundamentals (Routes, Pydantic Models)**
- REST API design principles
- Request/response modeling with Pydantic
- Input validation and error handling
- Dependency injection patterns

**Subject 6: FastAPI Advanced: Async, Background Tasks, Authentication**
- Asynchronous request handling
- Background task processing
- JWT-based authentication
- Security best practices

**Subject 7: Introduction to gRPC and Proto Design**
- REST vs gRPC comparison
- Protocol Buffer (.proto) file design
- gRPC service implementation
- Client-server communication patterns

### Section 3: Data Pipeline & Database Engineering
**Subject 8: Crawlee and Crawling Persian Websites (Python)**
- Web crawling fundamentals with Crawlee Python SDK
- Respectful crawling and robots.txt compliance
- Persian content extraction and processing
- Data deduplication and pagination handling

**Subject 9: Storing Data in PostgreSQL and Schema Design**
- Relational database design principles
- PostgreSQL schema creation and migrations
- SQLAlchemy ORM integration
- Database indexing strategies

**Subject 10: Building a Pipeline: Crawlee → DB → API**
- ETL (Extract, Transform, Load) pipeline design
- Data normalization and processing
- API endpoint creation for data access
- Error handling and logging in pipelines

**Subject 11: Serving with PostgreSQL and Optimization**
- Database connection pooling
- Query optimization and performance tuning
- Database backups and read replicas
- Monitoring and profiling database operations

### Section 4: Frontend & Integration
**Subject 12: Loveblae and AI-assisted Frontend Generation**
- AI-powered frontend development
- API integration with generated interfaces
- Modern UI/UX principles
- Rapid prototyping techniques

**Subject 13: Final Project and Evaluation**
- End-to-end application development
- System architecture design
- Project documentation and presentation
- Demo video creation and delivery

## Technology Stack

- **Version Control**: Git, GitHub
- **Containerization**: Docker, Docker Compose
- **Backend Framework**: FastAPI (Python)
- **Database**: PostgreSQL
- **ORM**: SQLAlchemy
- **Web Crawling**: Crawlee Python SDK
- **API Protocol**: REST, gRPC
- **Authentication**: JWT
- **Frontend Generation**: AI tools (Loveblae)
- **CI/CD**: GitHub Actions

## Assessment Structure

Each subject includes:
- **Goals**: Learning objectives for the topic
- **Lab**: Hands-on practical exercises
- **Assignment**: Individual or group projects
- **Resources**: Documentation and reference materials

## Final Project Requirements

The culminating project requires students to:
1. Design and implement a complete data pipeline (crawling → database → API)
2. Create an AI-generated frontend interface
3. Containerize the entire application
4. Provide comprehensive documentation
5. Deliver a 5-10 minute demo video

## Prerequisites

### Technical Requirements
- **Programming Knowledge**: Basic Python programming (variables, functions, classes)
- **Command Line**: Familiarity with terminal/command prompt operations
- **Web Concepts**: Understanding of HTTP, APIs, and web architecture
- **Git**: Basic understanding of version control (covered in Subject 1)
- **Docker**: Computer capable of running Docker containers

### Software Requirements
- **Python 3.8+**: Core programming language for the course
- **Git**: Version control system (installation covered in Subject 1)
- **Docker & Docker Compose**: Containerization platform
- **Text Editor/IDE**: VS Code, PyCharm, or similar (VS Code recommended)
- **Web Browser**: Modern browser for web development and testing

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 20GB free space for Docker images and project files
- **Internet Connection**: Required for downloading dependencies and accessing GitHub

### Optional but Recommended
- **GitHub Account**: For version control and collaboration
- **PostgreSQL Client**: For database work (pgAdmin, DBeaver, or similar)
- **API Testing Tool**: Postman, Insomnia, or similar for testing APIs

## Course Materials

All course materials, code examples, and assignments are available in this repository. Each subject folder contains:
- `README.md`: Subject overview, goals, and assignments
- `tutorials/`: Conceptual learning materials and reference guides
- `workshops/`: Hands-on practical exercises and labs
- `homeworks/`: Individual assignments and projects
- `assessments/`: Quiz questions, rubrics, and evaluation materials
- `resources/`: Additional documentation, useful links, and reference materials
- `installation/`: Setup guides and environment configuration

## Getting Started

1. **Clone this repository**
   ```bash
   git clone https://github.com/buqaen-courses/Special-Topics-04.git
   cd Special-Topics-04
   ```

2. **Set up your development environment**
   - Install Python 3.8+ and Docker
   - Follow installation guides in each subject's `installation/` folder
   - Set up Git and GitHub (see Subject 1)

3. **Follow subjects in numerical order**
   - Start with Subject 1: Git Basics & Project Workflows
   - Each subject builds on previous knowledge
   - Complete tutorials before attempting workshops

4. **Complete labs and assignments**
   - Read subject README.md for detailed instructions
   - Complete workshops for hands-on practice
   - Submit homeworks via GitHub fork and pull request
   - Use the provided assessment rubrics

5. **Use the final project to demonstrate your learning**
   - Build a complete full-stack application
   - Demonstrate all learned skills
   - Create documentation and demo video

## Homework Submission

All homework assignments must be submitted via GitHub using the fork-based workflow:

1. **Fork the course repository** to your GitHub account
2. **Create a Homeworks folder** in your fork with the proper subject structure
3. **Complete assignments** and commit your work
4. **Submit a pull request** with your completed homework
5. **Include required deliverables** (screenshots, documentation, code)

**Example structure for homework submission:**
```
Homeworks/
└── Subject-01-Git-basics/
    └── homework-02-branching-exercise/
        ├── README.md (your documentation)
        ├── screenshots/ (all required screenshots)
        └── repository-url.txt (link to your exercise repo)
```

See individual homework assignments for detailed submission requirements.

## Contributing

This course materials are maintained by the Computer Engineering department. Suggestions and improvements are welcome through GitHub issues and pull requests.

## License

Course materials are provided for educational purposes. Code examples may be used and modified for learning and academic projects.
