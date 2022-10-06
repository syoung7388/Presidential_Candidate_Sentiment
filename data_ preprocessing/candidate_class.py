import re
import pickle 
from collections import Counter



ljm = ['찢재명', '전과 4범', '전과4범', '둠재명','이재앙', '보급형 허겨영',  '보급형허겨영', '사이다 이재명', '형보수지', '혜경궁',
'김혜경', '이죄명', '이재명' ,'재명', '경기도지사', '죄명', '이후보','찢빠들', '재앙', '찢', '표미새', '혜경']
ysy = ['윤두환', '윤로남불', '쩍벌', '정치검사', '차차', '짜장', '핵관', '윤', '석열', '열', '짱깨', '윤후보', '무식', 
'김건희', '윤석렬', '썩열', '무당', '도리', '핵관', '짜왕', '쭉뻗', '오또케', '차차','춘장']

ljm_comment = []
ysy_comment = []
two_comment = []

def classification(lines):
		for info in lines:
				info = info.strip()
				info = info.split(',')
				if len(info) < 3: continue		
				l_flag = False
				y_flag = False
				for l in ljm:
						if l in info[2]:
								l_flag = True
								break
				for y in ysy:
						if y in info[2]:			
								y_flag = True
								break
				info = (info[0], info[1], info[2])
				if l_flag and y_flag:
						two_comment.append(info)
				elif y_flag:
						ysy_comment.append(info)
				elif l_flag:
						ljm_comment.append(info)



def datapreprocessing(comments):
		
		#3개 보다 많은 문장 제거
		arr = set()
		for comment in comments:
				string = ''
				for i in range(2, len(comment)):
						string += re.sub('[^ ㄱ-ㅣ가-힣]+', '', comment[i])
				if not string: continue
				arr.add((comment[0], comment[1], string))
		return arr

		
			
s = 0

for i in range(11): #11
		f =  open("/NasData/home/ksy/Basic/project/project_test/data/data_news_comment/1nd/comment"+str(i)+".txt", 'r')
		lines = f.readlines()
		s += len(lines)
		classification(lines)

print("total_comment:", s)

print("후보자 이름 있는 댓글 추출")

print("ljm:", len(ljm_comment))
print("ysy:", len(ysy_comment))
print("two:", len(two_comment))
print("중복제거")



ysy_comment = datapreprocessing(ysy_comment)
ljm_comment = datapreprocessing(ljm_comment)
two_comment = datapreprocessing(two_comment)






print("ljm:", len(ljm_comment))
print("ysy:", len(ysy_comment))
print("two:", len(two_comment))

ljm_comment, ysy_comment,two_comment = list(ljm_comment), list(ysy_comment), list(two_comment)


with open("textmining_data/ljm.txt","a", encoding="UTF-8") as f:
		for c in ljm_comment:
				f.write(c[0]+','+c[1]+','+ c[2]+'\n')
		


with open("textmining_data/ysy.txt","a", encoding="UTF-8") as f:
		for c in ysy_comment:
				f.write(c[0]+','+c[1]+','+ c[2]+'\n')
		

with open("textmining_data/two.txt","a", encoding="UTF-8") as f:
		for c in two_comment:
				f.write(c[0]+','+c[1]+','+ c[2]+'\n')
		




with open("textmining_data/ljm.p","wb") as f:
    pickle.dump(ljm_comment, f)

with open("textmining_data/ysy.p","wb") as f:
    pickle.dump(ysy_comment, f)

with open("textmining_data/two.p","wb") as f:
    pickle.dump(two_comment, f)


