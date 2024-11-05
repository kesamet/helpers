## Cloud Processing

1. Edge Layer:
   - Multiple cameras feeding into edge computing devices
   - Edge devices can perform initial preprocessing and filtering

2. Data Ingestion:
   - Kinesis Data Streams for real-time data ingestion
   - S3 input bucket for batch processing
   - Flexible to handle both real-time and batch workflows

3. Processing Layer:
   - Lambda functions trigger processing of new images
   - Model inference happens via SageMaker endpoint for scalability and management
   - SQS queue to handle processing results

4. Storage Layer:
   - S3 bucket for storing processed results
   - DynamoDB for metadata and classification results

5. Analysis & Monitoring:
   - EventBridge for alerts and automation
   - CloudWatch for monitoring and metrics
   - Lambda functions for custom analytics

6. Visualization:
   - Streamlit app running on EC2
   - Application Load Balancer (ALB) for load distribution
   - AWS WAF for web application security
   - Streamlit app directly accesses DynamoDB and S3 for data visualization


```mermaid
flowchart TD
    subgraph Edge["Edge Devices"]
        C1[Camera 1]
        C2[Camera 2]
        C3[Camera 3]
        EC[Edge Computing Device]
        C1 & C2 & C3 --> EC
    end

    subgraph Ingestion["Data Ingestion"]
        KDS[Kinesis Data Streams]
        S3I[S3 Input Bucket]
        EC --> KDS
        EC --> S3I
    end

    subgraph Processing["Image Processing"]
        Lambda1[Lambda Trigger]
        SageMaker[SageMaker Endpoint]
        SQS[SQS Queue]
        S3I --> Lambda1
        Lambda1 --> SageMaker
        SageMaker --> SQS
    end

    subgraph Storage["Storage Layer"]
        S3R[S3 Results Bucket]
        DDB[DynamoDB]
        SQS --> S3R
        SQS --> DDB
    end

    subgraph Analysis["Analysis & Monitoring"]
        EA[EventBridge Alerts]
        CW[CloudWatch Metrics]
        Lambda2[Lambda Analytics]
        DDB --> Lambda2
        Lambda2 --> EA
        Lambda2 --> CW
    end

    subgraph Visualization["Streamlit Application"]
        ALB[Application Load Balancer]
        EC2[EC2 Instance]
        S3R --> EC2
        DDB --> EC2
        EC2 --> ALB
    end

    subgraph Security["Security Layer"]
        WAF[AWS WAF]
        ALB --> WAF
    end

```
