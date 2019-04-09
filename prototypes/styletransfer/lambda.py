import json, boto3

REGION = 'eu-west-1' # region to launch instance.
AMI = "ami-0a961c5be1838f98e" # our custom AMI
INSTANCE_TYPE = 'p2.xlarge' # instance type to launch.

EC2 = boto3.client('ec2', region_name=REGION)

def format_response(message, status_code):
    return {
        'statusCode': str(status_code),
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            }
        }

def lambda_handler(event, context):

    body = json.loads(event['body'])
  
    init_script = f"""#!/bin/bash
    cd /home/ubuntu
    git clone https://github.com/gabrielelanaro/ml-prototypes.git
    source activate tensorflow_p36
    cd ml-prototypes
    python -m prototypes.styletransfer.app --address=0.0.0.0 --port=8000
    shutdown -h +10
    """
    
    instance = EC2.run_instances(
        ImageId=AMI,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, # required by boto, even though it's kinda obvious.
        MaxCount=1,
        KeyName="FraDeepLearn",
        InstanceInitiatedShutdownBehavior='terminate', # make shutdown in script terminate ec2
        UserData=init_script # file to run on instance init.
    )

    instance_id = instance['Instances'][0]['InstanceId']

    #inst = boto3.resource('ec2', region_name=REGION).Instance(instance_id)

    # Wait for the instance to enter the running state
    #inst.wait_until_running()

    # Reload the instance attributes
    #inst.load()
    #print(inst.public_dns_name)
    #print(type(inst.public_dns_name))

    return format_response(f"instance_id: {instance_id}", 200)