## CI/CD Pipeline

- Development:
    * GitHub repository for version control
    * Local testing environment for developers

- CI/CD Pipeline (GitHub Actions):
    * Testing Phase: Unit tests, integration tests, security scanning
    * Build Phase: Docker image creation and push to ECR
    * Deployment Phase: Infrastructure as Code with Terraform

- Production:
    * Auto Scaling Group for EC2 instances
    * Application Load Balancer for traffic distribution

- Monitoring:
    * CloudWatch for AWS metrics
    * Prometheus/Grafana for custom metrics


```mermaid
flowchart LR
    subgraph Development["Development Environment"]
        GR[GitHub Repository]
        LT[Local Testing]
        GR --> LT
    end

    subgraph CICD["CI/CD Pipeline"]
        direction TB
        GHA[GitHub Actions]
        
        subgraph Testing["Testing Phase"]
            UT[Unit Tests]
            IT[Integration Tests]
            SL[Security Linting]
            GHA --> UT & IT & SL
        end
        
        subgraph Build["Build Phase"]
            DB[Docker Build]
            ECR[Push to ECR]
            Testing --> DB
            DB --> ECR
        end
        
        subgraph Deploy["Deployment Phase"]
            TF[Terraform Apply]
            CD[CodeDeploy]
            ECR --> TF
            TF --> CD
        end
    end

    subgraph Production["Production Environment"]
        ASG[Auto Scaling Group]
        ALB[Application Load Balancer]
        CD --> ASG
        ASG --> ALB
    end

    subgraph Monitoring["Monitoring & Logging"]
        CW[CloudWatch]
        PRO[Prometheus]
        GRA[Grafana]
        ASG --> CW & PRO
        PRO --> GRA
    end

```
