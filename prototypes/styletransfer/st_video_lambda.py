import boto3, json
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    
    s3 = boto3.client('s3')
    
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = record['s3']['object']['key']
    
    response = s3.head_object(Bucket=bucket, Key=key)
    email = response['Metadata']['email']
    style = response['Metadata']['style']

    print(bucket)
    print(key)
    print(email)
    print(style)
    
    SENDER = "fra.pochetti@gmail.com"
    RECIPIENT = email
    AWS_REGION = "eu-west-1"
    SUBJECT = "VisualNeurons.com - your GIF has been ingested!"
    
    BODY_TEXT = ("VisualNeurons.com - your GIF has been ingested! \r\n"
                 "The purpose of this email is to confirm that we successfully ingested your file"
                 "and that we are currently processing it."
                )

    # The HTML body of the email.
    BODY_HTML = """
    <html>

    <head></head>

    <body>
        <h3>You just uploaded a GIF to S3. Congratulations!</h3>
        <p>The purpose of this email is to confirm that we successfully ingested your file and that we are currently processing it.
            <br>
            When our GPUs finish crunching your request, you will receive another email with the link to your Style-Transferred GIF.
            <br>
            This will happen in a hour or so. Thanks for your patience!
        </p>
    </body>

    </html>
                """            

    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses', region_name=AWS_REGION)

    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
    
    return