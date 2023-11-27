Báo cáo nhóm bao gồm các folder sau:
  1. static:  
	- asset: chứa các file css và javascript để xây dựng web
	- evaluation: kết quả đánh giá của từng model
	- feature: chứa các file Numpy array file - trích xuất đặc trưng của mỗi bức ảnh trong dataset
	- groundtruth: chứa kết quả truy xuất của từng ảnh đã được kiểm định
	- img: chứa các ảnh gốc trong dataset
	- logo: chứa các hình ảnh được sử dụng trong UI
	- uploaded: chứa ảnh query
  2. templates: chứa file html 
  3. Các file python chính dùng để để truy vấn hình ảnh: 
	- app
	- evalute
	- evaluting
	- feature_extractor
	- indexing 

Các bước để chạy model: 
Bước 1: Trích xuất feature image:
	  Tải Dataset Oxford Building và đưa vào folder img. Sau đó, chạy file indexing. 
Bước 2: Mở terminal tại folder "Image_Retrieval", sau đó chạy lệnh "flask run" 
Bước 3: Sử dụng localhost có được sau khi chạy bước 2 để thử và xem kết quả. 


Bạn có thể tham khảo demo tại [đây](https://drive.google.com/file/d/1FeUJPsvgnFQC97nEgTdwNBPyfo--pQGw/view?usp=drive_link)
