## 當openAI API 遇上 django rest framework
### 製作一個關於產生圖片的API

```
1.cd mysite 
2.pip install -r .\requirements.txt 
3.創建一個mysql db setting.py 更改 dbname and password
4.python .\manage.py makemigrations 
5.python .\manage.py migrate
6.python .\manage.py runserver
7.search (
  openai.organization = "insert your organization"
  openai.api_key = "insert your API-KEY"
)
8.modify 上述欄位，改為自己的資訊
9.postman測試 or 直接輸入 url路徑及可 method post and choose json or formdata

Example:
http://127.0.0.1:8000/api/openai-photo/create
describe: 描述圖案名稱
numbers: 產生圖片數量
fileName: 儲存檔案的名稱
{
    "describe":"a puppy is Vincent style",
    "numbers":3,
    "fileName":"colorful_airplant"
}

10.search photo url method get
http://127.0.0.1:8000/api/openai-photo/list 
11.圖片都存在於openAIDjango\mysite\media\artAI
```
## 以下是找尋API-KEY and organization 流程圖示

### 1.openai web 
![](https://github.com/leolee1204/openAIDjango/blob/5c82ed0c931625170f17ff1b6467fd51d146c2c1/temp/step-1.jpg)

### 2.search api-key
![](https://github.com/leolee1204/openAIDjango/blob/5c82ed0c931625170f17ff1b6467fd51d146c2c1/temp/step-2.jpg)

### 3.create api-key
![](https://github.com/leolee1204/openAIDjango/blob/5c82ed0c931625170f17ff1b6467fd51d146c2c1/temp/step-3.jpg)

### 4.search openai.organization
![](https://github.com/leolee1204/openAIDjango/blob/5c82ed0c931625170f17ff1b6467fd51d146c2c1/temp/step-4.jpg)

### 5.modify OpenAIPhotoCreateAPIView insert to your detail
![](https://github.com/leolee1204/openAIDjango/blob/5c82ed0c931625170f17ff1b6467fd51d146c2c1/temp/step-5.jpg)

### 6.create openPhoto
![](https://github.com/leolee1204/openAIDjango/blob/744f5cdc00311f338a3b4d43d566d19c24c43be1/temp/step-6.jpg)

### 7.openAIPhoto list
![](https://github.com/leolee1204/openAIDjango/blob/744f5cdc00311f338a3b4d43d566d19c24c43be1/temp/step-7.jpg)
