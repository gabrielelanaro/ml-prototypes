import json, boto3, botocore

# This file is to store a lambda handler which gets triggered by an Amazon API Gateway POST method.
# A lambda handler is a serverless python function executed each time the trigger is pulled.
# In this specific case, the function does the following:
# 1) spins up a p2.xlarge EC2 instance with a custom deep learning AMI
# 2) clones our transfer learning repo
# 3) launches the server side websocket
# 4) kills the EC2 after 10 minutes
# 5) returns to the browser the pulic DNS of the spinned up EC2 instance

REGION = "eu-west-1"  # region to launch instance.
AMI = "ami-0a961c5be1838f98e"  # our custom AMI
available_instances = [
    "g2.2xlarge",
    "g2.8xlarge",
    "g3.16xlarge",
    "g3.4xlarge",
    "g3.8xlarge",
    "g3s.xlarge",
    "p2.8xlarge",
    "p2.xlarge",
    "p3.2xlarge",
    "p3.8xlarge",
]

EC2 = boto3.client("ec2", region_name=REGION)


def format_response(message, status_code):
    return {
        "statusCode": str(status_code),
        "body": json.dumps(message),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }


def spin_up_ec2(ec2_object, instance_type, script):

    try:
        instance = ec2_object.run_instances(
            ImageId=AMI,
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            KeyName="FraDeepLearn",
            SecurityGroups=[
                "transfer-learning",
            ],
            InstanceInitiatedShutdownBehavior='terminate', # make shutdown in script terminate ec2
            UserData=script # file to run on instance init.
        )

    except botocore.exceptions.ClientError:
        instance = None

    return instance


def lambda_handler(event, context):

    body = json.loads(event["body"])

    init_script = f"""#!/bin/bash
    cd /home/ubuntu
    git clone https://github.com/gabrielelanaro/ml-prototypes.git
    source activate tensorflow_p36
    cd ml-prototypes
    shutdown -h +10
    python -m prototypes.styletransfer.app --address=0.0.0.0 --port=8000
    shutdown -h now
    """
    for INSTANCE_TYPE in available_instances:
        instance = spin_up_ec2(EC2, INSTANCE_TYPE, init_script)
        if instance:
            instance_id = instance["Instances"][0]["InstanceId"]
            break

    if not instance:
        return format_response(
            f"id: 'limit exceeded', type: 'limit exceeded', dns: 'limit exceeded'", 200
        )

    inst = boto3.resource("ec2", region_name=REGION).Instance(instance_id)
    inst.wait_until_running()
    inst.load()
    public_dns = inst.public_dns_name

    return format_response(
        f"id: {instance_id}, type: {INSTANCE_TYPE}, dns: {public_dns}", 200
    )
