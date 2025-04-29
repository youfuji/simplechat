import json
import os
import boto3
import re
import urllib.request
from botocore.exceptions import ClientError

# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# グローバル変数としてBedrockクライアントを初期化
bedrock_client = None

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

# FastAPIのURL
FASTAPI_URL = "https://dfaa-35-229-195-52.ngrok-free.app" 

def lambda_handler(event, context):
    try:
        # コンテキストから実行リージョンを取得し、クライアントを初期化
        global bedrock_client
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Initialized Bedrock client in region: {region}")
        
        print("Received event:", json.dumps(event))
        
        # Cognito認証されたユーザー情報
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        
        # 会話履歴を使用
        messages = conversation_history.copy()

        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Nova Liteモデル用のリクエストペイロードを構築
        # 会話履歴を含める
        bedrock_messages = []
        for msg in messages:
            role = msg["role"]
            content = [{"text": msg["content"]}]
            bedrock_messages.append({"role": role, "content": content})
        
        # invoke_model用のリクエストペイロード
        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 512,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        
        print("Calling Bedrock invoke_model API...")

        # invoke_model APIを呼び出し
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_payload),
            contentType="application/json"
        )
        
        # レスポンスを解析
        response_body = json.loads(response['body'].read())
        print("Bedrock response:", json.dumps(response_body, default=str))
        
        # 応答の検証
        if not response_body.get('output') or not response_body['output'].get('message') or not response_body['output']['message'].get('content'):
            raise Exception("No response content from the model")
        
        # アシスタント応答を取得
        assistant_response = response_body['output']['message']['content'][0]['text']
        
        # アシスタント応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # ここで FastAPI サーバーに送信する
        payload_to_fastapi = {
            "assistant_response": assistant_response,
            "conversation_history": messages
        }
        
        fastapi_request = urllib.request.Request(
            url=FASTAPI_URL,
            data=json.dumps(payload_to_fastapi).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(fastapi_request) as res:
                fastapi_response = res.read()
                print("FastAPI server responded:", fastapi_response.decode('utf-8'))
        except urllib.error.HTTPError as e:
            print(f"FastAPI HTTPError: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            print(f"FastAPI URLError: {e.reason}")

        # 最終的なレスポンス
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
    
    except Exception as error:
        print("Error:", str(error))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }