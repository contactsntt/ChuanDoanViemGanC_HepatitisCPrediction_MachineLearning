# ChuanDoanViemGanC_HepatitisCPrediction_MachineLearning
# Chuẩn Đoán Viêm Gan Siêu Vi C   (Hepatitis C Prediction)

---------------------------------------------------------------------------------
## MỤC LỤC:

I.	Tổng Quan:

   1.	Giới thiệu:
   
   2.	Phương pháp:	

      2.1.	Mô tả tập dữ liệu:	

      2.2. Tiền xử lý dữ liệu và tầm quan trọng:	
      
      2.3. Lựa chọn Model:	
      
      2.4. Đánh giá hiệu suất mô hình:	

3. Kết quả:	

4. Thảo luận:	

II.	Kết Luận:	

----------------------------------------------------------------------------------------

I.	Tổng Quan:

Tính Trừu tượng: Viêm gan C là một bệnh nhiễm trùng gan do siêu vi khuẩn viêm gan C (HCV) gây ra. Do khởi phát các triệu chứng muộn,  để chuẩn đoán sớm là rất khó. Dự đoán trước khi bị tổn thương gan là quá trình xuyên suốt. Mục tiêu chính của nghiên cứu này là sử dụng thuật toán để dự đoán bệnh dựa trên dữ liệu xét nghiệm để chuẩn đoán và điều trị bệnh ở giai đoạn đầu. Trong nghiên cứu này, các thuật toán sử dụng như  Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine …. 

Hiệu suất của các kỹ thuật này được so sánh về ma trận để xác định một phương pháp thích hợp để dự đoán bệnh. Phân tích SVM và XGBoost (với độ chính xác và AUC cao nhất trong số các thử nghiệm mẫu, >80%) có thể là công cụ hiệu quả cho các chuyên gia y tế sử dụng xét nghiệm thông thường và là dữ liệu để dự đoán viêm gan C.

1.	Giới thiệu:
   
Có rất nhiều bệnh ảnh hưởng đến gan của con người. Gan là một trong những bộ phận quan trọng. Bệnh viêm gan C do một loại vi-rút gây ra có thể gây tử vong nếu không được phát hiện. Nó có thể tiến từ từ và thậm chí gây ung thư. Trong một số trường hợp, nó có thể không hoạt động trong cơ thể thậm chí 10 – 20 năm. Một số người bị viêm gan C có thể chỉ bị trong thời gian ngắn, nhưng đối với hơn một nửa số người nhiễm vi-rút, nó sẽ tiến triển thành một bệnh nhiễm trùng mãn tính. Chỉ 30% bệnh nhân nhiễm vi rút viêm gan C tự hồi phục trong vòng sáu tháng, trong khi hầu hết bệnh nhân bị nhiễm vi rút mãn tính. Các vấn đề sức khỏe có khả năng gây tử vong có thể do viêm gan C mãn tính là xơ gan và ung thư gan. 

Thông thường, những người bị viêm gan C mãn tính không cảm thấy ốm hoặc biểu hiện bất kỳ triệu chứng nào và các triệu chứng chỉ xuất hiện khi bệnh đã tiến triển. Người bệnh có thể bị suy nhược, buồn ngủ và chóng mặt, tất cả đều có thể bị nhầm lẫn với kiệt sức do làm việc hoặc học tập. Đáng sợ hơn khi vắc-xin viêm gan C chưa có. Một xét nghiệm kháng thể được sử dụng để kiểm tra HCV. Những người dương tính với xét nghiệm kháng thể nên trải qua xét nghiệm axit nucleic (NAT) để tìm HCV. Một rào cản lớn trong quá trình điều trị liên tục là quy trình chẩn đoán hai bước, gây khó khăn cho bệnh nhân và dẫn đến bệnh nhân không được theo dõi thường xuyên. 

Nghiên cứu này nhằm mục đích chọn các thuật toán để dự đoán viêm gan C dựa trên thói quen và xét kết quả xét nghiệm.

2.	Phương pháp:

Về nghiên cứu này, trong bước đầu tiên, trước khi sử dụng các kỹ thuật học máy, mô hình của bộ dữ liệu, quá trình tiền xử lý và tầm quan trọng của tính năng được xem xét. Bước thứ hai là sử dụng và phát triển các mô hình học máy khác nhau. Cuối cùng, trong bước cuối cùng, hiệu suất của từng mô hình được đánh giá theo ma trận nhầm, độ chính xác, tỉ lệ chuẩn để dự đoán viêm gan C sau khi so sánh hiệu suất của các mô hình này. Các mô hình này được phát triển bằng các chương trình Python.

 <img src="https://private-user-images.githubusercontent.com/134685355/314562444-9ebb6673-8426-4a12-ac4c-18a5fa59a39a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5NDkyNzEsIm5iZiI6MTcxMDk0ODk3MSwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDQ0LTllYmI2NjczLTg0MjYtNGExMi1hYzRjLTE4YTVmYTU5YTM5YS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMFQxNTM2MTFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05NzAzOWM3NTk1OGVjYmE0MmY4MzM0N2FkMGQyNWU4MTRhZTBjMDEwNWEzMDVlM2I1MzZlYjI3NWY3ZjQ2NGM4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.gVKWW57T_qWJdgjBR7CrUkpaKo-bZk1ooq1zJL1V_Vo">

 Hình 1: Mô hình dự đoán viêm gan C.
 

2.1.	Mô tả tập dữ liệu:

Hai bộ dữ liệu đã được sử dụng trong nghiên cứu này. Bộ dữ liệu NHANES chứa dữ liệu của 148 cá nhân và được lấy từ Trung tâm Kiểm soát Dịch bệnh (CDC) của Hoa Kỳ như một phần của Khảo sát Kiểm tra Sức khỏe và Dinh dưỡng Quốc gia (NHANES). Tính năng liên quan đến viêm gan C đã được chọn từ bộ dữ liệu này: 	

Age	ALB	ALP	ALT	AST	BIL	CHE	CHOL    CREA   GGT     PROT 

Mục tiêu sẽ phân loại các cá nhân thành những người mắc bệnh hoặc không mắc viêm gan C, bao gồm cả sự tiến triển của nó thành xơ hóa và xơ gan.

2.2. Tiền xử lý dữ liệu và tầm quan trọng:

Vì chất lượng dữ liệu là yếu tố quan trọng cần cân nhắc trong quy trình khai thác dữ liệu để dự đoán và chẩn đoán bệnh, nên một quy trình làm sạch dữ liệu đã được sử dụng để làm cho bộ dữ liệu chính xác hơn. Một số thuộc tính trong bộ dữ liệu này có một số giá trị bị thiếu và trùng lặp. Trong nghiên cứu này, phương pháp cắt bỏ trung vị được sử dụng để điền vào các giá trị còn thiếu trong bộ dữ liệu. 

Phương pháp này được chọn vì nó là một kỹ thuật đơn giản và được sử dụng rộng rãi để xử lý dữ liệu bị thiếu, đặc biệt khi các giá trị bị thiếu được cho là bị thiếu một cách ngẫu nhiên. Giá trị trung bình là một thước đo mạnh mẽ về xu không bị ảnh hưởng bởi các giá trị ngoại lệ và ít có khả năng đưa ra sai lệch trong phân tích so với các phương pháp khác, chẳng hạn như quy nạp trung bình. Nghiên cứu trước đây cũng đã chỉ ra rằng phép loại bỏ trung vị có thể hoạt động tốt khi so sánh với các phương pháp loại bỏ khác về mặt giảm sai lệch và tăng độ chính xác của phân tích. Ngoài ra, phương pháp cắt bỏ trung bình đã được sử dụng trong các nghiên cứu y học khác nhau để xử lý dữ liệu bị thiếu. Do đó, dựa trên tính đơn giản và mạnh mẽ của phương pháp, cũng như thành công của nó trong nghiên cứu trước đây, phương pháp quy nạp trung vị được coi là phù hợp.

Tiêu chuẩn hóa dữ liệu đã được áp dụng cho các bộ dữ liệu sau khi xử lý dữ liệu bị thiếu, xóa các dữ liệu trùng lặp và chuyển đổi một số tính năng chuỗi thành số, bước tiếp theo là các kỹ thuật đánh giá các đặc điểm đầu vào theo mức độ chúng có thể dự đoán một biến mục tiêu. Trong nghiên cứu này, các phương pháp nhúng đã được sử dụng để tính toán tầm quan trọng của quá trình chuyển đổi chuỗi thành số.

Trong học máy, phương pháp nhúng là một kỹ thuật lựa phù hợp nhất trong quá trình đào tạo mô hình. Điều này đạt được bằng cách kết hợp chuyển đổi vào thuật toán được sử dụng để đào tạo mô hình, thay vì thực hiện chuyển đổi như một bước tiền xử lý riêng biệt.

Các phương pháp nhúng hoạt động bằng cách đánh giá tầm quan trọng của việc chuyển đổi trong quá trình đào tạo mô hình và chỉ chọn những phương hướng phù hợp vào độ chính xác của mô hình. Điều này thường đạt được bằng cách gán trọng số hoặc hệ số cho từng tính năng dựa trên chuyển đổi của nó trong mô hình.

2.3. Lựa chọn Model:

Trong nghiên cứu này, sử dụng các thuật toán học máy khác nhau để dự đoán bệnh viêm gan C bằng cách sử dụng dữ liệu xét nghiệm để chẩn đoán và điều trị bệnh ở giai đoạn đầu. Để đạt được mục tiêu này, đã chọn các thuật toán học máy, bao gồm Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machine, dựa trên hiệu quả đã được chứng minh của chúng trong chẩn đoán y khoa. Đối với mỗi thuật toán, sử dụng các siêu tham số để đảm bảo độ chính xác tối đa. Ví dụ: chúng tôi đã sử dụng C = 1.0, gamma = tỷ lệ, hàm nhân = hàm cơ sở radial (RBF) cho SVM và mục tiêu = nhị phân: logistic, learning_rate = 0.2 cho XGBoost. Tương tự, dụng hàm kích hoạt = relu, hidden layer = 3, bộ giải = lbfgs cho Logistic Regression. Sự lựa chọn của những siêu tham số đóng vai trò quan trọng trong việc đạt được độ chính xác cao của các mô hình.

2.4. Đánh giá hiệu suất mô hình:

Các số liệu thống kê khác nhau đã được sử dụng để đánh giá các thuật toán học máy đã phát triển. Thông tin này bao gồm Kết quả xác thực đúng (TP), Kết quả xác thực sai (FP), Kết quả âm tính  (TN) và Kết quả dương tính (FP) với phép đo hiệu suất, đó là ma trận, độ chính xác,…

Xác suất có bệnh khi kết quả dương tính được gọi là giá trị dự đoán dương tính và khi kết quả âm tính, xác suất không bệnh được gọi là giá trị dự đoán âm tính. Ma trận được sử dụng để cung cấp một bản tóm tắt các kết quả dự đoán.

Độ chính xác được xác định bằng mức độ gần đúng của kết quả thu được với giá trị thực của phép đo. Vì lý do này, độ chính xác cũng có thể được tính như sau:

độ chính xác = (TP + TN)/(TP + TN + FP + FN)

3. Kết quả:
   
Tổng cộng, 148 cá nhân (59 nữ và 89 nam) và 615 (238 nữ và 377 nam) đã được đưa vào bộ dữ liệu NHANES và UCI tương ứng. Bộ dữ liệu NHANES và bộ dữ liệu UCI sẽ được lọc như hình bên dưới:

<img src="https://private-user-images.githubusercontent.com/134685355/314562453-bcbfc888-17aa-40aa-98b4-9dab393b23f1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDUzLWJjYmZjODg4LTE3YWEtNDBhYS05OGI0LTlkYWIzOTNiMjNmMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zYzY0NzkwMzIxZTcyZDAwODQ5Nzg2NTQ3MGNmZTRjZDkxM2M4YmNmNDRlMDM5ODc0NThmYTAyNzdkZTcxNGVlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.CAC3gv_gVFqcLZILXeKPUALNy7jPKqxIXqxWHArpVDg">
 
Hình 2: ALB: albumin; ALP: Alkaline phosphatase; ALT: alanine aminotransferase; AST: aspartate aminotransferase; BIL: total bilirubin; CHE: serum cholinesterase; PROT: total protein; CHOL: cholesterol; GGT: y-glutamyl transferase; CREA:creatinine blood (CREA); BIL: bilirubin.

<img src="https://private-user-images.githubusercontent.com/134685355/314562457-aa705ab1-a1c3-4539-8aab-b524fe341273.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDU3LWFhNzA1YWIxLWExYzMtNDUzOS04YWFiLWI1MjRmZTM0MTI3My5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zMDkzODEzYzkyZTFhYzhiMmM3MzljMDk4ZmNiY2RlMTFhNmMxZTNiMjExYjRkMzY3NjdjNzc3OTEzZTdlZDExJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.BwIUF8ntXmYZRDvLe9zCQEF3hmdOoztI3mpYuGKPB8I">
 
Hình 3: Bản đồ nhiệt cho mối quan hệ giữa các thuộc tính khác nhau dựa trên bộ dữ liệu.

Sau khi xử lý dữ liệu bị thiếu, loại bỏ các dữ liệu trùng lặp và chuyển đổi một số tính năng chuỗi thành số, tiêu chuẩn hóa dữ liệu đã được áp dụng cho các bộ dữ liệu. Phân tích tương quan đã được thực hiện sau đó. Bản đồ nhiệt được minh họa trong Hình ở trên được sử dụng để hiển thị mối tương quan giữa các tính năng trong bộ dữ liệu này. Sau đó, giá trị của tầm quan trọng của tính năng được tính toán để xếp hạng các tính năng theo phương pháp nhúng. Thứ hạng tầm quan trọng của các tính năng trong bộ dữ liệu được hiển thị trong Hình bên dưới. Theo phân tích này, trọng số các tính năng này, giới tính có tầm quan trọng thấp nhất và AST có tầm quan trọng nhất trong bộ dữ liệu này.

 <img src="https://private-user-images.githubusercontent.com/134685355/314562462-46919b6b-e365-4d57-8c9b-f9a5477e8a47.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDYyLTQ2OTE5YjZiLWUzNjUtNGQ1Ny04YzliLWY5YTU0NzdlOGE0Ny5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01ZGY5MWUzNzcyZjliNWZhZWE2ZTMyM2I0MDc4ZmY4MGM2YmUzZTRkZDQwZTEwOGQ0MzJkZWQ2ZWViNDZlNzFmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.7wEZvHTUJUtPK9aZzqFujvQmVJxUGxeYxyaPhdk-D20">
 
Hình 4: Tầm quan trọng của tính năng dựa trên bộ dữ liệu.

Số liệu cho các thuật toán này được bao gồm trong Tài liệu bổ sung. Việc tính toán sai số tuyệt đối trung bình (MAE) cho cả tập dữ liệu huấn luyện và kiểm tra là một phương pháp phổ biến trong học máy để tránh bị quá mức. Xảy ra khi một mô hình quá phức tạp và không khớp với dữ liệu huấn luyện, dẫn đến hiệu suất kém trên dữ liệu mới. Bằng cách tính toán MAE cho cả tập dữ liệu huấn luyện và kiểm tra, hiệu suất của mô hình có thể được đánh giá và so sánh để xác định xem mô hình có khớp với dữ liệu huấn luyện hay không.

<img src="https://private-user-images.githubusercontent.com/134685355/314562465-fc67870d-f434-41f4-8e8b-5fdbe48ad43e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDY1LWZjNjc4NzBkLWY0MzQtNDFmNC04ZThiLTVmZGJlNDhhZDQzZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lYjY5NWIxMDMyZWMxYWVjZGU4NTFhMjIxMWRjNGIyYmQyZjgyZGMwMGMxOTg0MmNjZmM2MDljN2M0NjllYmQwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.C87wAagqf6i6M_nFoLHsiUbz5x-sEGKbaNwLCaqIL7g">
 
Hình 5: các mô hình và vẽ ma trận trong ô con tương ứng.

 <img src="https://private-user-images.githubusercontent.com/134685355/314562472-28ec8cf0-18cd-498e-b933-7c66d349c92c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDcyLTI4ZWM4Y2YwLTE4Y2QtNDk4ZS1iOTMzLTdjNjZkMzQ5YzkyYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05YWEyNmI0OWY0MGVmNTkzMzkxNjg3OGNjZmIyYTcwZWZmNWU5ZGI1NDZhNmFlODc5YWU5Nzg3YTY5ZjU3ODEzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.xD6ndeBGQ9zj9AABwwwZBas7whnVGvGv9lKbbO-m1Q4">
 
Hình 6: Tỉ lệ độ chính xác.

<img src="https://private-user-images.githubusercontent.com/134685355/314562477-ae3a2420-b2c6-45d8-8748-60c688964b30.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDc3LWFlM2EyNDIwLWIyYzYtNDVkOC04NzQ4LTYwYzY4ODk2NGIzMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iNGZmMjI5NDY4NDIzN2NmZjU3ZGVhZjcxYjcyZGFjMjYwMzc4OWRjNmU1YmQ0NDAxZTY4ZDVhNjhmY2JlOTNjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Aup25Q-OTsZAMnyx8BM1y1mGYB766CTmibgfYsegho8">
 
Hình 7:Tầm quan trọng của tính năng.

4. Thảo luận:
   
Dựa trên các kết quả thử nghiệm, kỹ thuật SVM và XGBoost có thể được sử dụng làm công cụ hiệu quả sử dụng dữ liệu xét nghiệm thông thường và không tốn kém để dự đoán bệnh viêm gan C. Qua nhiều năm, các kỹ thuật học máy đã được sử dụng trong dự đoán bệnh. Kết quả cũng xác nhận rằng thuật toán XGBoost là một mô hình chính xác cho dự đoán viêm gan C. 
Mặc dù biểu đồ bản đồ nhiệt cho thấy tuổi tác và giới tính có mối tương quan yếu với các đặc điểm khác, nhưng có một số bằng chứng và nghiên cứu cho thấy chúng là những yếu tố quan trọng liên quan đến bệnh viêm gan C. Một tổng quan hệ thống và phân tích tổng hợp để đánh giá và phân tích sự khác biệt về giới trong bệnh viêm gan C liên quan đến tỷ lệ nhiễm vi-rút. Phát hiện ra rằng tỷ lệ dương tính với HCV RNA ở nam giới trưởng thành cao hơn đáng kể so với nữ giới. Phụ nữ cũng tiến triển với tốc độ chậm hơn nam giới nếu họ bị nhiễm mãn tính.

 <img src="https://private-user-images.githubusercontent.com/134685355/314562485-ba02e607-fdaf-4078-a900-62ec2be57a8c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDg1LWJhMDJlNjA3LWZkYWYtNDA3OC1hOTAwLTYyZWMyYmU1N2E4Yy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mNTM2OWZiNTFjNjVhMDU2ODUwMDg3ZTBmMThlZDAyZGRmOWMwNTY2ZjNkMDI2NTQxMDhjZmYyMjJjM2Q5MGVlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.qh_nkCozTU8UqLKP1BNTqrxIA7NkECmF1GiRcmAlHbc">
 
Hình 8:Phân phối các kết quả xét nghiệm máu khác nhau theo danh mục.

 <img src="https://private-user-images.githubusercontent.com/134685355/314562491-7af19956-8891-42f7-8513-1a9c8e69fe05.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDkxLTdhZjE5OTU2LTg4OTEtNDJmNy04NTEzLTFhOWM4ZTY5ZmUwNS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jMjkwZmY2Njk0OWNkMzMyM2FjODMyOWVjMThiNWRlYzgyYjAyNTJlMzJlY2Y5NzFiNTNkOWE2MGRhODg0OThmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.xGLroqRGI8udM0BqcJAD52pCjMwy4IgtjoMTS70UkGs">
 
Hình 9: tỷ lệ dương tính với HCV RNA ở nam giới & nữ giới.

Tuổi tác là tình trạng dẫn đến mất dần khả năng duy trì cân bằng nội môi của cơ thể, khiến tuổi tác trở thành một yếu tố nguy cơ đáng kể đối với các bệnh mãn tính. Các nghiên cứu đã chỉ ra rằng bệnh viêm gan C có thể gây tổn thương các cơ quan khác nhau, đặc biệt là gan và tổn thương này sẽ trầm trọng hơn theo tuổi tác. So với một người lớn tuổi, một người trẻ tuổi có thể sống sót trong nhiều năm với tổn thương gan tối thiểu hoặc không có. Tuy nhiên, khi các cá nhân già đi, hệ thống miễn dịch của họ suy yếu và tốc độ xơ hóa thường tăng nhanh. Mặc dù các bản đồ nhiệt cho thấy mối tương quan thấp với các biến khác, nhưng việc loại trừ hai biến này có thể dẫn đến sai lệch trong nghiên cứu và việc bao gồm hai biến này có thể được coi là một trong những lợi thế.
Nhược điểm của nghiên cứu này là không phải tất cả các tính năng quan trọng đều có thể được giữ lại trong bộ dữ liệu, đây là điển hình của phân tích dữ liệu thứ cấp.

 <img src="https://private-user-images.githubusercontent.com/134685355/314562496-e823a146-fd75-4119-921b-26c1e7116317.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA5ODQ0NTIsIm5iZiI6MTcxMDk4NDE1MiwicGF0aCI6Ii8xMzQ2ODUzNTUvMzE0NTYyNDk2LWU4MjNhMTQ2LWZkNzUtNDExOS05MjFiLTI2YzFlNzExNjMxNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMyMVQwMTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xNDRjYzMxMWNjNTBiNDZmNTRiMzJiNmEyMzQ4YjA3YmEwNDQzYjE4NTUxODc3ZTY4ZWM2NGQyNWI0M2YyNmJhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.wSntK4j3pWhDWN-4yE4tvk1KqMLlEkYGilb9K0U4bDU">
 
Hình 10: tuổi và loại sau khi chuyển đổi.

II.	Kết Luận:

Tóm lại, nghiên cứu này nhằm mục đích sử dụng các kỹ thuật học máy khác nhau để dự đoán bệnh viêm gan C dựa trên dữ liệu xét nghiệm máu định kỳ. Kết quả chứng minh rằng các kỹ thuật SVM và XGBoost có hiệu quả trong chẩn đoán viêm gan C ở giai đoạn đầu với độ chính xác và AUC cao (>80%). Phân tích trên cũng xác định các tính năng quan trọng như ALT, ALB, ALP, AST, TBIL, PROT, CHOL, CHE, GGT, CREA, giới tính và độ tuổi có thể được sử dụng trong các kỹ thuật này. Tuy nhiên, nghiên cứu còn một số hạn chế, bao gồm việc sử dụng các bộ dữ liệu hạn chế, thiếu dữ liệu lâm sàng và không có thử nghiệm lâm sàng. Trong tương lai, mong muốn kết hợp các tính năng bổ sung liên quan đến viêm gan C để phát triển các kỹ thuật máy học hiệu quả và đáng tin cậy hơn. Nhìn chung, nghiên cứu cho kết quả đầy mong đợi trong việc phát hiện và chẩn đoán sớm bệnh viêm gan C bằng các kỹ thuật học máy, điều này cuối cùng có thể cải thiện kết quả điều trị của người mắc bệnh.
