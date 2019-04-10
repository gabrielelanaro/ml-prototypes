import json, boto3

# This file is to store a lambda handler which gets triggered by an Amazon API Gateway POST method.
# A lambda handler is a serverless python function executed each time the trigger is pulled.
# In this specific case, the function does the following:
# 1) spins up a p2.xlarge EC2 instance with a custom deep learning AMI
# 2) clones our transfer learning repo
# 3) launches the server side websocket
# 4) kills the EC2 after 10 minutes
# 5) returns to the browser the instance_id of the spinned up EC2 instance

REGION = "eu-west-1"  # region to launch instance.
AMI = "ami-0a961c5be1838f98e"  # our custom AMI
INSTANCE_TYPE = "p2.xlarge"  # instance type to launch.

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


def lambda_handler(event, context):

    body = json.loads(event["body"])

    init_script = f"""#!/bin/bash
    cd /home/ubuntu
    git clone https://github.com/gabrielelanaro/ml-prototypes.git
    source activate tensorflow_p36
    cd ml-prototypes
    # We shut down the system in 10 minutes if the server doesn't return earlier than that
    shutdown -h +10
    python -m prototypes.styletransfer.app --address=0.0.0.0 --port=8000
    shutdown -h now
    """

    instance = EC2.run_instances(
        ImageId=AMI,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,  # required by boto, even though it's kinda obvious.
        MaxCount=1,
        KeyName="FraDeepLearn",
        InstanceInitiatedShutdownBehavior="terminate",  # make shutdown in script terminate ec2
        UserData=init_script,  # file to run on instance init.
    )

    instance_id = instance["Instances"][0]["InstanceId"]

    # Lines 58-61 are used to retrieve the public DNS of the EC2 instance
    # Lambda has spun up. This attribute is made available only when the
    # machine switches to its RUNNING state, so not at boot time.
    # This section is commented as even if the code works as intented,
    # for unclear reasons, Lambda does not return the expected output

    # inst = boto3.resource('ec2', region_name=REGION).Instance(instance_id)
    # inst.wait_until_running()
    # inst.load()
    # public_dns = inst.public_dns_name

    return format_response(f"instance_id: {instance_id}", 200)
