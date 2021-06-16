import concurrent.futures
import requests
import threading
import time
import json,base64,urllib
from termcolor import colored
import numpy as np


thread_local = threading.local()

quit = False

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

def config():
    # Opening JSON file 
    with open('config.txt',encoding='utf-8') as json_file: 
        data = json.load(json_file) 
    
        # for reading nested data [0] represents 
        # the index value of the list 

        return data



def download_site(proxy):
    # session = get_session()
    # url = "https://www.google.com"
    # with session.get(url) as response:
    #     print(f"Read {len(response.content)} from {url}")
    # count = 0
    success = 0
    fail = 0
    # key_value = 0
    configList = config()
    # sumSuccess = successSum()
    encoding = urllib.parse.quote(configList['game_name'])
    ip_addresses = proxy
    
        
    while True:
        # print (success)

        # startProcess = time.time()
        # print(key_value)
        # print("Success : ",success)
        a_fileJson = open("status.json", "r")
        json_objectJson = json.load(a_fileJson)
        a_fileJson.close()

        # print(json_objectJson['success'])
        if json_objectJson['success'] >= configList['rang']:
            return 1
            break
        checksum = get_checksum(ip_addresses, encoding)
        if checksum:
            # print("[INFO]ID Image : ",checksum)
            print(colored("[INFO]ID Image : ", "cyan"),colored(checksum, 'cyan'))

            # print(sum_time)
            # print(ip_addresses)
            # while True:
            image_contect = get_image(checksum, ip_addresses, encoding)
            # if image_contect:
            #     # print(image_contect)
            #     break
        # while True:
            image_base64 = base64.b64encode(image_contect).decode("utf-8")
            res = requests.post(configList['server_api']+'/api/predict/', json={
                'image': image_base64,
                'api_key' : configList['api_key']
            })
        
            if res:
                answer = res.json()['answer']
                # print("[INFO]Detection Time : "+str(res.elapsed.total_seconds()))
                print(colored("[INFO]Detection Time : ", "cyan"),colored(res.elapsed.total_seconds(), 'cyan'))
                # break
            
            # print("[INFO]Captcha Answer : "+answer)
            print(colored("[INFO]Captcha Answer: ", "cyan"),colored(answer, 'cyan'))
            if answer == 'NoApiKey':
                print('กรุณาติดต่อเจ้าของ API เพื่อซื้อ API KEY')
                # break
            else:
            # while True:
                start = time.time()
                status = post_image(checksum, answer, ip_addresses, configList, encoding)
                end = time.time()
                # if status:
                #     break
                # if status['success']:
                #     print(colored("[STATUS]", 'green'),colored(status, 'green'))
                # else:
                #     if status['used']:
                #         print(colored("[STATUS]", 'yellow'),colored(status, 'yellow'))
                #     else:
                #         print(colored("[STATUS]", 'red'),colored(status, 'red'))
                        
                # print(status)
                print(colored("[INFO]CALL POST TIME : ", "cyan"),colored(end - start, 'cyan'))
                # print("[INFO]CALL POST TIME : ",end - start)


                if status['success']:
                    success += 1
                    print(colored("[STATUS]", 'green'),colored(status, 'green'))
                    
                    # totalSuccess = int(sumSuccess)+1
                    # print(totalSuccess)
                    # return 1
                    # file = open('status.txt', 'w') 
  
                    # # Data to be written 
                    # data = str(totalSuccess)
                    
                    # # Writing to file 
                    # file.write(data) 
                    
                    # # Closing file 
                    # file.close() 
                    # key_value += 1
                    # open file in read mode 
                        
                    # open file in write mode 
                    # with open("status.txt", "w") as f: 
                        
                    #     for line in data : 
                            
                    #         # condition for data to be deleted 
                    #         if line.strip("\n") != data :  
                    #             f.write(line) 
                    break
                    # time.sleep(sum_time)
                # elif status['success'] == False and status['used'] == True:
                #     key_value += 1
                else:
                    # print(status['used'])
                    if status['used']:
                        print(colored("[STATUS]", 'yellow'),colored(status, 'yellow'))
                    #     ip_addresses = proxy
                    #     return ip_addresses
                    # else:
                        time.sleep(status['wait'])
                        # print("Done.........................................................")
                        # key_value += 1
                        
                    else:
                        print(colored("[STATUS]", 'red'),colored(status, 'red'))
                        with open('fail//{}.jpg'.format(checksum), 'wb') as File:
                            File.write(image_contect)
                    # fail += 1
                    # ip_addresses = proxy
                    # return ip_addresses
                    # if status['used'] == False:
                    #     with open('fail//{}.jpg'.format(checksum), 'wb') as File:
                    #         File.write(image_contect)
                        
                    # time.sleep(sum_time)
                        
                            # with open('fail\\{}.jpg'.format(checksum), 'wb') as File:
                            #     File.write(image_contect)
                    # time.sleep(2)
            # endProcess = time.time()
            # print("All Process : ",endProcess - startProcess)
            # break
        else:
            print("[INFO]IP : "+ip_addresses+ " Not Running")
            # key_value += 1
            

            # # open file in read mode 
            # with open("proxydict/p_list.txt", "r") as f: 
                
            #     # read data line by line  
            #     data = f.readlines() 
                
            # # open file in write mode 
            # with open("proxydict/p_list.txt", "w") as f: 
                
            #     for line in data : 
                    
            #         # condition for data to be deleted 
            #         if line.strip("\n") != ip_addresses :  
            #             f.write(line) 
            
            break
    



def download_all_sites(sites,configList):
    # print(sites)
    workers = configList['max_workers']
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        x = executor.map(download_site, sites)
        # print(str(i))
        
        # print(x)
        for i in x:
            # print("asdasdaskdasjkfnaskfjkasfjkasfjkasjkfnasfjkasfnaksnfafnasjhfasjhfjasfbasjhfb : ",str(i))
            return str(i)

        # first = True
        # print("[ ", end="")
        # for i in executor.map(download_site, sites):
        #     if first:
        #         first = False
        #     else:
        #         print(" , ", end="")

        #     print(str(i), end="")
        # print("]", end="")
        # if x:
        #     finals = []
        #     for value in x:
        #         finals.append(value)
        #     print(finals)
        #         # sites = value
        #     finals_list = []
        #     for target_list in finals:
        #         # print(target_list)
        #         if target_list is not None:
        #             # print(target_list)
        #             finals_list.append(target_list)
        #             # x = executor.map(download_site, target_list)
        #     executor.map(download_site, finals_list)



def get_checksum(ip_addresses, encoding):
    try:
        
        proxies = {
            "https": "https://{}/".format(ip_addresses),
            "http": "http://{}/".format(ip_addresses)
        }
        url = 'http://playserver.co/index.php/Vote/ajax_getpic/'+encoding
        r = requests.post(url, 
            headers={
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'th,en;q=0.9,en-GB;q=0.8,en-US;q=0.7',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Origin': 'http://playserver.in.th',
                # 'Pragma': 'no-cache',
                'Host': 'playserver.co',
                'Referer': 'http://playserver.in.th/index.php/Vote/prokud/'+encoding,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36 Edg/84.0.522.52'
            },
            timeout=10,
            proxies=proxies
        )
        if r.status_code == 200:
            # with open('proxydict/p_list.txt','r') as fp:
            #     vp = fp.read().splitlines()
            #     if ip_addresses not in vp:
            #         with open('proxydict/p_list.txt','a') as wf:
            #             wf.write(ip_addresses+"\n")
            return str(r.json()['checksum'])
        else:
            return False
    except Exception as e:
        return False

def get_image(checksum,ip_addresses, encoding):
    try:
        
        proxies = {
            "https": "https://{}/".format(ip_addresses),
            "http": "http://{}/".format(ip_addresses)
        }
        url2 = 'http://playserver.co/index.php/VoteGetImage/{}'.format(checksum)
        r2 = requests.get(url2, 
            headers={
                
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'th,en;q=0.9,en-GB;q=0.8,en-US;q=0.7',
                'Cache-Control': 'no-cache',
                # 'Pragma': 'no-cache',
                'Host': 'playserver.co',
                'Referer': 'http://playserver.in.th/index.php/Vote/prokud/+encoding',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36 Edg/84.0.522.52'
            },
            timeout=10,
            proxies=proxies
        )
        if r2.status_code == 200:
            return r2.content
        else:
            return False
    except Exception as e:
        return False

def post_image(checksum, answer, ip_addresses, configList, encoding ):
    

    try:
        
        proxies = {
            "https": "https://{}/".format(ip_addresses),
            "http": "http://{}/".format(ip_addresses)
        }

        # r = requests.get(url, proxies=proxies, verify=False)
        url3 = 'http://playserver.co/index.php/Vote/ajax_submitpic/'+encoding
        r2 = requests.post(url3, 
            headers={
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'Origin': 'http://playserver.in.th',
                'Host': 'playserver.co',
                'Referer': 'http://playserver.in.th/index.php/Vote/prokud/'+encoding,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36 Edg/84.0.522.52'
            },
            timeout=15,
            data={
                'server_id': configList['server_id'],
                'captcha': f'{answer}',
                'gameid': configList['gameid'],
                'checksum': f'{checksum}'
            },
            proxies=proxies, 
            verify=False
            

        )
        if r2.status_code == 200:
            status = r2.json()
            if status['success']:
                # total = successRead()
                # # successSum(total)
                # totalSum = total['success']+1
                # # print(totalSum)
                # with open("status.txt", "w") as jsonFile:
                #     # json.dump(totalSum, jsonFile['success'])
                #     print(jsonFile)
                # # totalSum = int(total)+1
                a_file = open("status.json", "r")
                json_object = json.load(a_file)
                a_file.close()
                # print(json_object)
                print(colored("[INFO] SUCCESS : ", 'blue'),colored(json_object['success']+1, 'blue'))

                json_object["success"] = json_object["success"]+1

                a_file = open("status.json", "w")

                json.dump(json_object, a_file)

                a_file.close()
                

            return r2.json()
        else:
            return False
    except Exception as e:
        return False


if __name__ == "__main__":


    configList = config()
    proxys1 = []
    with open('list_proxy_private.txt', 'r') as filehandle:
        proxys1 = [current_place.rstrip() for current_place in filehandle.readlines()]

    a_file = open("status.json", "r")
    json_object = json.load(a_file)
    a_file.close()

    json_object["success"] = 0
    a_file = open("status.json", "w")
    json.dump(json_object, a_file)
    a_file.close()
    
    
    while True:
        
        # arr = np.concatenate((proxys1))
        start_time = time.time()
        x = download_all_sites(proxys1,configList)
        if x == "1":
            break
        duration = time.time() - start_time
        print(f"Call {len(proxys1)} in {duration} seconds")
            
    
    
    
        
    
    
    