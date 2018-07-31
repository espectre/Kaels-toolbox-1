import json
import qiniu
import requests


def qiniu_auth(ak, sk):
    return qiniu.Auth(ak, sk)


def qiniu_mac_auth(ak, sk):
    return qiniu.QiniuMacAuth(ak, sk)


# def qiniu_request(api, data_url, auth, **kwargs):
#     payload = {"data": {"uri": data_url}}
#     req, resp = qiniu.http._post_with_auth(api, data=payload, auth=auth)
#     if status_code == '200':
#         buffer_json = json.dumps(str(req))
#     else:
#         buffer_json = json.loads('{"error":"Status code is not 200"}')
#     return buffer_json


def _payload_wrapper(data_url, data_type='image'):
    '''
    '''
    payload = dict()
    if data_type == 'image':
        payload['data'] = dict()
        payload['data']['uri'] = data_url 
    elif data_type == 'video':
        payload['data'] = dict()
        payload['data']['uri'] = data_url 
        payload['params'] = dict()
        payload['params']['async'] = False 
        payload['params']['vframe'] = dict()
        payload['params']['vframe']['interval'] = 3
        payload['params']['vframe']['mode'] = 0
        payload['ops'] = list() 
        payload['ops'].append({'op': 'pulp'}) 
    else:
        print('ERROR: unsupported data type: {}'.format(data_type))
    return payload


def post_request(auth, conf, data_url, proto='http://'):
    '''
    post one single request to atserving api
    '''
    host = conf.get('params', 'host') 
    query = conf.get('params', 'query') 
    data_type = conf.get('params', 'data_type')
    payload = _payload_wrapper(data_url, data_type=data_type)
    token = auth.token_of_request(
        method='POST',
        host=host,
        url=query,
        content_type='application/json',
        qheaders='',
        body=json.dumps(payload)
    )
    headers = {"Content-Type": "application/json", "Authorization": "Qiniu {}".format(token)}
    return requests.post(proto+host+query, headers=headers, data=json.dumps(payload))
