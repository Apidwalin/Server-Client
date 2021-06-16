import requests
import time
import base64
from requests_toolbelt.utils import dump
import json
import os , sys, traceback
from numpy import random
import urllib.parse

def config():
    # Opening JSON file 
    with open('config.txt',encoding='utf-8') as json_file: 
        data = json.load(json_file) 
    
        # for reading nested data [0] represents 
        # the index value of the list 

        return data

def get_checksum(encoding):
    try:
        
        # print(encoding)
        url = 'http://playserver.co/index.php/Vote/ajax_getpic/'+encoding
        r = requests.post(url, 
            headers={
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'th,en;q=0.9,en-GB;q=0.8,en-US;q=0.7',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Origin': 'http://playserver.in.th',
                'Pragma': 'no-cache',
                'Host': 'playserver.co',
                'Referer': 'http://playserver.in.th/index.php/Vote/prokud/'+encoding,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36 Edg/84.0.522.52'
            },
            timeout=5
        )
        if r.status_code == 200:
            return str(r.json()['checksum'])
        else:
            return False
    except Exception as e:
        return False

def get_image(checksum, encoding):
    try:
        url2 = 'http://playserver.co/index.php/VoteGetImage/{}'.format(checksum)
        r2 = requests.get(url2, 
            headers={
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'th,en;q=0.9,en-GB;q=0.8,en-US;q=0.7',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Host': 'playserver.co',
                'Referer': 'http://playserver.in.th/index.php/Vote/prokud/'+encoding,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36 Edg/84.0.522.52'
            },
            timeout=5
        )
        if r2.status_code == 200:
            return r2.content
        else:
            return False
    except Exception as e:
        return False


def post_image(checksum, answer, configList, encoding):
    

    try:
        url3 = 'http://playserver.co/index.php/Vote/ajax_submitpic/'+encoding
        r2 = requests.post(url3, 
            headers={
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'Origin': 'http://playserver.in.th',
                'Host': 'playserver.co',
                'Referer': 'http://playserver.in.th/index.php/Vote/prokud/'+encoding,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36 Edg/84.0.522.52'
            },
            timeout=5,
            data={
                'server_id': configList['server_id'],
                'captcha': f'{answer}',
                'gameid': configList['gameid'],
                'checksum': f'{checksum}'
            }

        )
        if r2.status_code == 200:
            return r2.json()
        else:
            return False
    except Exception as e:
        return False

def main():
    count = 0
    success = 0
    fail = 0
    # x = input('Enter your API KEY:')
    configList = config()
    encoding = urllib.parse.quote(configList['game_name'])
    while True:
        while True:
            print("Success : ",success)
            checksum = get_checksum(encoding)
            if checksum:
                break
        print(checksum)
        # print(configList)
        while True:
            image_contect = get_image(checksum, encoding)
            if image_contect:
                break
        while True:
            image_base64 = base64.b64encode(image_contect).decode("utf-8")
            res = requests.post(configList['server_api']+'/api/predict/', json={
                'image': image_base64,
                'api_key' : configList['api_key']
            })
            if res:
                answer = res.json()['answer']
                print("Detection Time : "+str(res.elapsed.total_seconds()))
                break
        
        print("Captcha Answer : "+answer)
        if answer == 'NoApiKey':
            print(res.json()['text'])
            break
        else:
            while True:
                status = post_image(checksum, answer, configList, encoding)
                if status:
                    break

            print(status)
            if status['success']:
                success += 1
                time.sleep(61)
            else:
                fail += 1
                
                    # with open('fail\\{}.jpg'.format(checksum), 'wb') as File:
                    #     File.write(image_contect)
            # time.sleep(2)
        count += 1     
    # print(count, success, fail)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("{0}".format("-" * 60))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("{0}".format("-" * 60))
    finally:
        input("press any key for exit")
        sys.exit()