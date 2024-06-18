import boto3

iam = boto3.client('iam')

response = iam.create_user(
    UserName='data-analyst'
)

access_key_response = iam.create_access_key(
    UserName='data-analyst'
)

print(access_key_response['AccessKey']['AccessKeyId'])
print(access_key_response['AccessKey']['SecretAccessKey'])
