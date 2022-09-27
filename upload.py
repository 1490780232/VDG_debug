# -*- coding=utf-8
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos.cos_exception import CosClientError, CosServiceError
import sys
import logging

# 正常情况日志级别使用INFO，需要定位时可以修改为DEBUG，此时SDK会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# 1. 设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在CosConfig中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
secret_id = 'AKIDKUO1NSGn32kBLEqBF3aqpXy81dx0mHVR'     # 替换为用户的 SecretId，请登录访问管理控制台进行查看和管理，https://console.cloud.tencent.com/cam/capi
secret_key = 'h7G9l58uPdaGemdDVXIxLRHp66pHSM0O'   # 替换为用户的 SecretKey，请登录访问管理控制台进行查看和管理，https://console.cloud.tencent.com/cam/capi
region = 'ap-nanjing'      # 替换为用户的 region，已创建桶归属的region可以在控制台查看，https://console.cloud.tencent.com/cos5/bucket
                           # COS支持的所有region列表参见https://cloud.tencent.com/document/product/436/6224
token = None               # 如果使用永久密钥不需要填入token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见https://cloud.tencent.com/document/product/436/14048
scheme = 'https'           # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Scheme=scheme)
client = CosS3Client(config)

# 使用高级接口上传一次，不重试，此时没有使用断点续传的功能
# response = client.upload_file(
#     Bucket='data-1257243463',
#     Key='exampleobject',
#     LocalFilePath='/home/lzy/00_reid_models/RCEL/RESL_code/examples/data.zip',
#     EnableMD5=False,
#     progress_callback=None
# )

# 使用高级接口断点续传，失败重试时不会上传已成功的分块(这里重试10次)
for i in range(0, 10):
    try:
        response = client.upload_file(
        Bucket='data-1257243463',
        Key='market_train_fpn_view_features_final.zip',
        LocalFilePath='/home/lzy/Generate/re-id/00_inversion_person/encoder4editing/generate_images/market_train_fpn_view_features_final.zip',)
        break
    except CosClientError or CosServiceError as e:
        print(e)