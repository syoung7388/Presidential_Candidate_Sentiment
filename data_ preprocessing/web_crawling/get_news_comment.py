# -*- coding: cp949 -*-
# module import
import requests
import urllib
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import time
import numpy as np
import csv
import torch
import os
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import argparse




parser = argparse.ArgumentParser(description='Naver_Comment')
parser.add_argument('--candidate', type=str, default='윤석열', help='candidate')
parser.add_argument('--startpage', type=int, default='1', help='startpage')
parser.add_argument('--endpage', type=int, default='50', help='endpage')
parser.add_argument('--date', type = str, default = '_0201_0308_', help = 'date')
args = parser.parse_args()

# 변수 설정

search_QUERY = urllib.parse.urlencode({'query':args.candidate}, encoding='utf-8')
URL = f"https://search.naver.com/search.naver?where=news&query=%EB%8C%80%EC%84%A0&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds=2022.02.01&de=2022.03.09&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Ar%2Cp%3Afrom20220201to20220309&is_sug_officeid=0"

LINK_PAT = "naver"
comment_url = "m_view=1&"


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# driver 설정vi
s = Service("/NasData/home/ksy/Basic/project/project_test/data/driver_99/chromedriver")
driver = webdriver.Chrome(service = s, chrome_options = chrome_options)

make_file = open("news_comment"+"_"+str(args.startpage)+'_'+str(args.endpage)+"_"+args.date+".txt", 'w', encoding="UTF-8")
make_tot_file = open("news_comments_NAVER_"+str(args.startpage)+'_'+str(args.endpage)+"_"+args.date+".txt", mode="w", encoding="UTF-8")
make_file.close()
make_tot_file.close()



total_comment = 0

		
# 검색결과 내 링크 찾기 : news.naver.com으로 시작하는 모든 링크 반환
def get_news_links(startpage, endpage, link_pattern):
		links = set()
		for page in range(startpage, endpage+1):
				print(f"Scrapping page : {page}", end = " ") # 확인용
				req = requests.get(f"{URL}&start={page}"); print(req.status_code)
				soup = BeautifulSoup(req.text, 'lxml')
				results = soup.find_all('a', {'href': re.compile(link_pattern), 'class':'info'})
				for result in results:
						links.add(result['href'])

		print(f"총 {len(links)}개의 뉴스 링크를 찾았습니다.") # 확인용
		return list(links)


# 한 페이지 별로 필요한 정보 스크레이핑
def extract_info(url, wait_time=1,delay_time =0.3):



		try:
				driver.implicitly_wait(wait_time)
				driver.get(url)
				# 댓글 창 있으면 다 내리기
			
						
						
				print(url)
				for _ in range(30):
						try:
								more_comments = driver.find_element(By.XPATH, '//*[@id="cbox_module"]/div[2]/div[9]/a')
								more_comments.click()
								time.sleep(delay_time)
				
						except:
								break
		
						
				# html 페이지 읽어오기
				html = driver.page_source
				
				soup = BeautifulSoup(html, 'lxml')
				     
				
				# 출처
				site = soup.find('h1').find("span").get_text(strip=True)
				# 기사 제목
				title = soup.find('h3', {'id':'articleTitle'}).get_text(strip=True)
				
				# 작성 시간
				article_time = soup.find('span', {'class':'t11'}).get_text(strip=True)
				
				# 언론사
				press = soup.find('div', {'class':"press_logo"}).find('a').find('img')['title']
				# 댓글 수
				
				
				total_com = soup.find("span", {"class" : "u_cbox_info_txt"}).get_text()
				
				
				# 댓글 작성자
				nicks = soup.find_all("span", {"class":"u_cbox_nick"})
				nicks = [nick.text for nick in nicks]
				
				# 댓글 날짜
				dates = soup.find_all("span", {"class":"u_cbox_date"})
				dates = [date.text for date in dates]
				
				# 댓글 내용
				contents = soup.find_all("span", {"class":"u_cbox_contents"})
				contents = [content.text for content in contents]
				
				
				f = open("news_comment"+"_"+str(args.startpage)+'_'+str(args.endpage)+"_"+args.date+".txt", 'a', encoding="UTF-8")
				reply = []
				for i in range(len(contents)):
						reply.append({'nickname':nicks[i],
						              'date':dates[i],
						              'contents':contents[i]})
				
						f.write(nicks[i]+','+dates[i]+','+ contents[i]+'\n')
				
				f.close()
				print("완료", len(reply)) # 확인용
				print("save comment number:", total_comment)
				if reply:
						print("comment:", reply[0])
				
				return {'site':site, 'title':title, 'article_time':article_time, 'press':press, 'total_comments':total_com, 'reply_content':reply}
		except:
				print(url)
				return {}	



							
					
# 각 페이지 돌면서 스크레이핑
def extract_comments(links):
		contents = []
	
		for link in links:
				content = extract_info(f"{link}&m_view=1") # 각각의 링크에 대해 extract_info 함수 호출
				if len(content) == 0: continue
				contents.append(content) # extract_info의 결과로 나오는 dict 저장
		return contents


# main 함수
def main():
		news_links = get_news_links(args.startpage, args.endpage, LINK_PAT) 
		result = extract_comments(news_links)
		return result

# 출력 결과 저장 함수
def save_to_file(lst):
		print('save_to_file')
		file = open("news_comments_NAVER_"+str(args.startpage)+'_'+str(args.endpage)+"_"+args.date+".txt", mode="a", encoding="UTF-8")
		writer = csv.writer(file) 
		writer.writerow(['site', 'title', 'article_time', 'press', 'total_comments', 'contents'])
		for result in lst:
				writer.writerow(list(result.values()))
		file.close()

# 함수 실행
NAVER_RESULT = main()
save_to_file(NAVER_RESULT)