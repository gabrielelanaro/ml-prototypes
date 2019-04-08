import json
import boto3

REGION = 'eu-west-1' # region to launch instance.
AMI = 'ami-047bb4163c506cd98' # Amazon Linux AMI as per https://aws.amazon.com/amazon-linux-ami/
INSTANCE_TYPE = 'p2.xlarge' # instance type to launch.

EC2 = boto3.client('ec2', region_name=REGION)

def lambda_handler(event, context):

    body = json.loads(event['body'])
    message = body['data']

    # run 2 commands, then shut instance down after 2 minute
    init_script = """#!/bin/bash
    yum update -y
    yum install -y httpd24
    shutdown -h +2"""

    instance = EC2.run_instances(
        ImageId=AMI,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, # required by boto, even though it's kinda obvious.
        MaxCount=1,
        InstanceInitiatedShutdownBehavior='terminate', # make shutdown in script terminate ec2
        UserData=init_script # file to run on instance init.
    )

    instance_id = instance['Instances'][0]['InstanceId']

    return {
        'statusCode': 200,
        'body': json.dumps({"data": message,
                           "instance_id": instance_id})
    }