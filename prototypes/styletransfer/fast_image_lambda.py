import base64, os, boto3, ast, json 

endpoint = 'kandinsky'

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
    style = body["style"].replace(".png", "")
    
    if style != "Kand2": return format_response(f"Sorry, {style} is not supported yet!", 400)
    
    image = base64.b64decode(body['data']) 
    runtime = boto3.Session().client(service_name='sagemaker-runtime', region_name='eu-west-1')
    response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='image/jpeg', Body=image)
    r = json.loads(response['Body'].read().decode())
    
    return format_response(r['prediction'], 200)