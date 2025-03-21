# AWS Bedrock Credentials Guide

This guide provides detailed instructions on how to obtain and configure AWS Bedrock credentials for use with the LLM Dialog Manager.

## Prerequisites

- An AWS account
- Basic understanding of AWS IAM (Identity and Access Management)
- Admin access to your AWS account (or ability to create IAM users and policies)

## Getting AWS Bedrock Credentials

### 1. Create an IAM User

1. Log in to the [AWS Management Console](https://console.aws.amazon.com)
2. Navigate to IAM (Identity and Access Management) service
3. Click on "Users" in the left navigation panel
4. Click "Create user"
5. Enter a name for the user (e.g., "bedrock-api-user")
6. Under "Select AWS access type", choose "Access key - Programmatic access"
7. Click "Next: Permissions"

### 2. Attach Permissions to the IAM User

For full access to Bedrock services:

1. On the permissions page, choose "Attach policies directly"
2. Search for and select "AmazonBedrockFullAccess"
3. Click "Next: Tags" (optional), then "Next: Review"
4. Click "Create user"

For more restricted access (recommended for production), create a custom policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:GetFoundationModel"
      ],
      "Resource": "*"
    }
  ]
}
```

### 3. Enable Model Access in Bedrock

Before you can use any models:

1. Navigate to Amazon Bedrock in the AWS console
2. Go to "Model access" in the left navigation
3. Request access to the specific models you want to use
4. Wait for approval (usually immediate for AWS foundation models)

### 4. Create and Retrieve Access Keys

1. After user creation, you'll see the "Success" page displaying your credentials
2. **IMPORTANT**: This is the ONLY time you'll see the Secret Access Key
3. Copy both the Access Key ID and Secret Access Key
4. Alternatively, download the CSV file containing your credentials

If you missed this window, you'll need to create new access keys:

1. Go to the IAM console and select your user
2. Select the "Security credentials" tab
3. Scroll to "Access keys" and click "Create access key"
4. Choose "Other" as the use case
5. Click "Create access key"
6. Copy or download your new credentials

### 5. Session Tokens (Optional)

Session tokens are temporary credentials, typically used with:
- Multi-factor authentication (MFA) 
- Temporary elevated access
- Federated identities

To obtain a session token using AWS CLI:

```bash
aws sts get-session-token --duration-seconds 3600
```

This will return an Access Key ID, Secret Access Key, and Session Token.

## Configuring Your Environment

Add the following to your `.env` file:

```
# AWS Bedrock
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_SESSION_TOKEN=your_session_token_if_using_temporary_credentials

# AWS Bedrock Region
AWS_REGION=us-west-2  # Or your preferred region where Bedrock is available
```

## Available AWS Regions for Bedrock

As of 2025, Amazon Bedrock is available in the following regions:
- us-east-1 (N. Virginia)
- us-west-2 (Oregon)
- ap-northeast-1 (Tokyo)
- ap-southeast-1 (Singapore)
- eu-central-1 (Frankfurt)
- eu-west-1 (Ireland)

Check the [AWS Region Table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/) for the most current availability.

## Security Best Practices

1. **Never commit credentials** to version control
2. Regularly rotate access keys (recommended every 90 days)
3. Use the principle of least privilege - only grant necessary permissions
4. Consider using temporary credentials with limited time windows
5. Set up CloudTrail for logging and monitoring Bedrock API calls
6. Implement IAM roles instead of access keys when possible

## Troubleshooting

### Common Issues

1. **"Access Denied" errors**:
   - Verify the IAM user has the correct permissions
   - Ensure you've enabled access to the specific models you're trying to use

2. **"Region not supported" errors**:
   - Confirm Bedrock is available in your configured AWS region
   - Some models may only be available in specific regions

3. **Credential validation errors**:
   - Check for typos in your credentials
   - Make sure your access keys are active (not deleted or deactivated)
   - Verify the AWS_REGION is set correctly

For additional help, consult the [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/).
