# LENS: Thử nghiệm phân đoạn tổn thương da trên CPU với loss vùng biên và TTA trong điều kiện thiếu dữ liệu được gán nhãn

## Tóm tắt

Nghiên cứu thử nghiệm ban đầu này đánh giá các cách phân đoạn tổn thương da bằng mô hình nhẹ khi số lượng ảnh đã được gán nhãn còn hạn chế. Chúng tôi lấy U-Net gọn nhẹ, huấn luyện bằng binary cross-entropy kết hợp soft Dice, làm mô hình cơ sở. Sáu phương án khác được đem ra so sánh gồm: thêm trọng số cho vùng biên dựa trên khoảng cách; thêm trọng số biên dựa trên mức không chắc chắn của mô hình; TTA bốn góc nhìn rồi lấy trung bình xác suất; gộp dự đoán trong không gian logit theo cách ít bị ảnh hưởng bởi một góc nhìn bất thường; và U-Net rộng hơn nhưng chỉ chạy suy luận một lần. Thí nghiệm dùng một tập con ISIC 2016 được chuẩn bị trên máy, gồm 140 cặp ảnh–mặt nạ cho huấn luyện, 40 cặp cho validation và 60 cặp cho test. Trong lần chạy thử trên CPU, chương trình chọn một nhóm cố định gồm 80 ảnh làm nguồn dữ liệu huấn luyện, 20 ảnh cho validation và 30 ảnh cho test; tất cả được thu nhỏ về 64 × 64. Mỗi cấu hình chạy với ba seed giống nhau để có thể so sánh công bằng và được huấn luyện trong hai epoch. Hai mức thí nghiệm lần lượt dùng 20 ảnh và 40 ảnh đã được gán nhãn, tương ứng 25% và 50% nhóm 80 ảnh ban đầu. Khi dùng 40 ảnh, U-Net cơ sở đạt Dice trung bình 0,4001; TTA kiểu robust-logit đạt 0,4043; còn mô hình rộng hơn đạt 0,3586. Khi dùng 20 ảnh, các kết quả lần lượt là 0,3883; 0,3922 và 0,3004. Các phiên bản có thêm loss vùng biên chỉ lệch dưới 0,0003 Dice so với mô hình cơ sở. Không phép so sánh nào cho thấy mức cải thiện đủ rõ về mặt thống kê. Vì vậy, kết quả hiện tại chưa cho thấy loss vùng biên, các cách TTA hoặc việc tăng độ rộng mô hình giúp phân đoạn tốt hơn trong cấu hình đã thử. Một nghiên cứu đầy đủ hơn cần kiểm tra xem phần loss được thêm vào có thật sự tác động đến quá trình học hay không, đồng thời huấn luyện lâu hơn, dùng nhiều dữ liệu hơn và lặp lại thí nghiệm độc lập.

**Từ khóa:** xử lý ảnh y khoa, phân đoạn ảnh, tổn thương da, U-Net, học với ít dữ liệu được gán nhãn, loss vùng biên, tăng cường dữ liệu tại thời điểm kiểm thử.

## 1. Giới thiệu

Phân đoạn tổn thương da là việc xác định chính xác vùng tổn thương trên ảnh soi da. Đây là một bài toán quan trọng trong xử lý ảnh y khoa, nhưng việc vẽ mặt nạ cho từng ảnh tốn nhiều thời gian và cần người có chuyên môn. Đường biên của tổn thương thường khó nhận ra do độ tương phản thấp, lông tóc che khuất, ánh sáng thay đổi, nhiễu từ thiết bị hoặc hình dạng không đều. Vì vậy, cần có những mô hình vẫn học tốt khi chỉ có một số lượng nhỏ ảnh đã được chuyên gia vẽ mặt nạ và không đòi hỏi quá nhiều tài nguyên tính toán.

U-Net và các mô hình encoder–decoder tương tự thường được dùng làm mốc so sánh trong phân đoạn ảnh y khoa [siddique2021unet][wang2022medical]. Loss chú trọng vùng biên cố gắng làm cho mô hình quan tâm nhiều hơn đến đường viền tổn thương, vì loss tính trên toàn vùng có thể chưa phạt đủ mạnh các lỗi nhỏ nằm sát đường biên [jurdi2021highlevel]. Một cách khác là TTA: cùng một ảnh được lật hoặc xoay theo nhiều hướng, mô hình dự đoán từng phiên bản, sau đó các kết quả được đưa về cùng chiều và gộp lại. TTA có thể làm dự đoán ổn định hơn, nhưng hiệu quả còn tùy vào cách biến đổi và cách gộp kết quả [ashraf2022melanoma]. Mức khác nhau giữa các dự đoán cũng có thể dùng như một dấu hiệu cho thấy mô hình chưa chắc chắn. Tuy nhiên, dấu hiệu này không thể tự động được xem là một phép đo uncertainty đã được kiểm chứng [abdar2021review][mehrtash2020confidence].

Nghiên cứu tập trung vào một câu hỏi cụ thể: trong điều kiện máy tính và dữ liệu đều hạn chế, việc thêm loss cho vùng biên hoặc gộp dự đoán từ bốn góc nhìn có giúp mặt nạ dự đoán khớp với mặt nạ thật hơn U-Net gọn nhẹ hay không? Các cách này cũng được so sánh với một U-Net rộng hơn nhưng chỉ suy luận một lần. Kết quả chỉ phản ánh thí nghiệm trên dữ liệu ảnh; nghiên cứu không đưa ra kết luận về chẩn đoán, điều trị, độ an toàn hay khả năng dùng trong bệnh viện.

## 2. Phương pháp

### 2.1. Dữ liệu và quy trình thực nghiệm

Thí nghiệm sử dụng các cặp ảnh–mặt nạ có nguồn gốc từ ISIC 2016. Chương trình chuẩn bị dữ liệu tạo ra 140 cặp cho huấn luyện, 40 cặp cho validation và 60 cặp cho test. Trong lần chạy thử, mã nguồn chọn 80 ảnh làm nguồn dữ liệu huấn luyện, 20 ảnh cho validation và 30 ảnh cho test. Ảnh và mặt nạ được đưa về kích thước 64 × 64 pixel. Từ nhóm 80 ảnh ban đầu, hai mức thí nghiệm lần lượt dùng 20 ảnh và 40 ảnh để huấn luyện, tương ứng với tỷ lệ 25% và 50%. Đây là một tập con nhỏ, không phải kết quả đánh giá trên toàn bộ bộ dữ liệu của cuộc thi.

Mỗi phương pháp được chạy với ba seed: 0, 1 và 2. Mô hình được huấn luyện trên CPU trong tối đa hai epoch bằng AdamW, batch size 8, cosine learning-rate schedule và early stopping. U-Net gọn nhẹ dùng số kênh 4, 8, 16 và 32 ở các tầng. Phiên bản rộng hơn được chọn bằng phần đo thời gian có sẵn trong mã thí nghiệm. Xác suất từ 0,5 trở lên được xem là vùng tổn thương để tạo mặt nạ nhị phân.

### 2.2. Các cấu hình được so sánh

Bảy cấu hình thực nghiệm gồm:

1. **C-UNet:** U-Net gọn nhẹ, sử dụng entropy chéo nhị phân kết hợp soft Dice và suy luận một góc nhìn.
2. **MP-TTA:** C-UNet kết hợp TTA bốn góc nhìn bằng cách lấy trung bình xác suất.
3. **UDB:** U-Net gọn nhẹ với loss tăng trọng số cho vùng gần đường biên.
4. **UB-TTA:** UDB kết hợp TTA bằng trung bình xác suất.
5. **UGDB:** U-Net gọn nhẹ với loss vùng biên có trọng số thay đổi theo mức không chắc chắn của mô hình.
6. **RL-TTA:** C-UNet kết hợp cơ chế đồng thuận bền vững trong không gian logit từ bốn góc nhìn.
7. **W-UNet:** U-Net rộng hơn, suy luận một góc nhìn.

Chỉ số chính là Dice trung bình trên từng ảnh. Dice càng cao thì vùng dự đoán càng trùng với mặt nạ thật. Thí nghiệm còn lưu IoU, foreground-balanced Brier score, lesion-centered Brier score, negative log-likelihood, expected calibration error (ECE), boundary-band Brier score, boundary F-score, calibration slope và intercept, HD95 cùng normalized surface Dice. Vì đây chỉ là lần chạy thử nhỏ với ba seed, kết quả được dùng để xem xu hướng và mức dao động, chưa đủ để khẳng định một phương pháp tốt hơn về mặt thống kê.

## 3. Kết quả

### 3.1. Kết quả Dice

| Phương pháp | Huấn luyện bằng 20 ảnh (25%), Dice trung bình ± độ lệch chuẩn | Huấn luyện bằng 40 ảnh (50%), Dice trung bình ± độ lệch chuẩn |
|---|---:|---:|
| C-UNet | 0,3883 ± 0,0543 | 0,4001 ± 0,0490 |
| MP-TTA | 0,3854 ± 0,0737 | 0,3974 ± 0,0658 |
| UDB | 0,3883 ± 0,0543 | 0,4002 ± 0,0490 |
| UB-TTA | 0,3854 ± 0,0737 | 0,3975 ± 0,0657 |
| UGDB | 0,3884 ± 0,0542 | 0,4003 ± 0,0489 |
| RL-TTA | **0,3922 ± 0,0639** | **0,4043 ± 0,0557** |
| W-UNet | 0,3004 ± 0,0756 | 0,3586 ± 0,0507 |

TTA kiểu robust-logit đạt Dice trung bình cao nhất trong cả hai trường hợp, nhưng chỉ cao hơn C-UNet 0,0039 khi huấn luyện bằng 20 ảnh và 0,0042 khi huấn luyện bằng 40 ảnh. Các phép kiểm định theo từng cặp seed chưa cho thấy khác biệt có ý nghĩa thống kê; khoảng tin cậy đều đi qua 0. Trong khi đó, TTA lấy trung bình xác suất làm Dice giảm khoảng 0,0029 và 0,0027.

Các mô hình có thêm loss vùng biên cho kết quả gần như giống mô hình cơ sở. So với C-UNet, Dice của UDB chỉ thay đổi khoảng +0,00002 ở mức 25% và +0,00011 ở mức 50%; UGDB thay đổi khoảng +0,00004 và +0,00016. Những con số này quá nhỏ so với mức dao động giữa các seed. Điều đó không có nghĩa mọi loại loss vùng biên đều vô ích; nó chỉ cho thấy cách cài đặt và thiết lập được chạy trong thí nghiệm này chưa làm thay đổi kết quả một cách đáng kể.

Trái với giả thuyết ban đầu về năng lực mô hình, W-UNet kém hơn C-UNet 0,0879 Dice ở mức 25% và 0,0415 ở mức 50%. Với chỉ hai epoch, việc tăng độ rộng có thể làm tăng khó khăn tối ưu hóa hoặc độ biến thiên mà không có đủ số lần cập nhật để tạo ra lợi ích về năng lực biểu diễn.

### 3.2. Diễn giải các chỉ số xác suất và đường biên

Tệp kết quả có đầy đủ các chỉ số phụ cho từng seed và từng cấu hình. Chúng giúp kiểm tra lại thí nghiệm, nhưng số ảnh ít và thời gian huấn luyện ngắn nên chưa thể kết luận chắc chắn về độ tin cậy của xác suất hay chất lượng đường biên. Brier score phản ánh cả độ khớp của xác suất lẫn khả năng phân biệt hai lớp; ECE thay đổi theo cách chia bin; boundary F-score và HD95 có thể dao động mạnh khi dự đoán quá kém hoặc không có vùng dương tính. Vì vậy, không thể dựa vào một chỉ số riêng lẻ để nói rằng mô hình đã đủ đáng tin cậy cho y tế [muller2022guideline][maierhein2024metrics].

## 4. Thảo luận

Kết quả chính của lần chạy này là chưa có phương án nào cải thiện rõ ràng so với U-Net gọn nhẹ chỉ suy luận một lần. TTA kiểu robust-logit tốt hơn một chút, còn TTA lấy trung bình xác suất lại kém hơn một chút. Điều này cho thấy cách gộp các dự đoán có thể ảnh hưởng đến kết quả, nhưng ba seed là quá ít để biết chênh lệch đó có ổn định hay chỉ do ngẫu nhiên.

Việc C-UNet, UDB và UGDB cho kết quả gần như giống nhau là điểm cần chú ý. Thêm một loss phức tạp không đồng nghĩa với việc loss đó đã thật sự ảnh hưởng đến quá trình học. Lần thử tiếp theo nên ghi lại giá trị loss vùng biên, độ lớn gradient của nó so với loss chính, mức thay đổi của trọng số uncertainty, sự khác nhau giữa các mặt nạ dự đoán và sự thay đổi trong tham số mô hình. Đồng thời, mô hình cần được huấn luyện lâu hơn và đánh giá trực tiếp bằng các chỉ số khoảng cách đường biên.

Không nên từ kết quả này mà kết luận rằng mô hình rộng luôn kém trong phân đoạn ảnh y khoa. Tất cả mô hình chỉ được học hai epoch trên một tập ảnh nhỏ và đã bị giảm độ phân giải. Mạng rộng hơn có thể cần learning rate khác, cách chống overfitting khác hoặc nhiều bước cập nhật hơn. Kết quả ở đây chỉ đúng với lần chạy thử hiện tại và cho thấy rằng tăng số kênh chưa chắc đem lại lợi ích nếu thời gian huấn luyện quá ngắn.

## 5. Hạn chế

- Nghiên cứu sử dụng một tập con nhỏ của ISIC 2016, không phải đánh giá đầy đủ trên toàn bộ chuẩn chính thức.
- Ảnh được giảm xuống 64 × 64, làm mất nhiều chi tiết đường biên nhỏ.
- Thời gian huấn luyện tối đa chỉ hai epoch nên nguy cơ mô hình chưa học đủ là rất cao.
- Chỉ có ba seed và một cách chia tập dữ liệu được đánh giá.
- Các điều kiện 25% và 50% là tỷ lệ của tập phát triển 80 ảnh, không phải tỷ lệ của toàn bộ kho huấn luyện ISIC.
- Chưa thử trên dữ liệu từ nguồn khác, chưa kiểm tra khi loại máy chụp hoặc điều kiện chụp thay đổi, chưa phân tích thông tin theo từng bệnh nhân và chưa có đánh giá của bác sĩ.
- Nhiều chỉ số được xem xét cùng lúc, nhưng số lần chạy chưa đủ lớn và chưa có bước điều chỉnh thống kê cho việc kiểm tra nhiều giả thuyết.
- Thời gian chạy trên CPU không phải phép đo độ trễ được chuẩn hóa cho triển khai thực tế.

## 6. Kết luận

Trong lần chạy thử bằng CPU, U-Net gọn nhẹ đạt Dice trung bình 0,3883 khi được huấn luyện bằng 20 ảnh và 0,4001 khi được huấn luyện bằng 40 ảnh. TTA kiểu robust-logit làm Dice tăng khoảng 0,004, nhưng mức tăng này chưa đủ rõ để khẳng định bằng thống kê. Hai cách thêm loss vùng biên hầu như không làm kết quả thay đổi, còn mô hình rộng hơn lại kém hơn khi chỉ được huấn luyện hai epoch. Nghiên cứu tiếp theo nên dùng bộ dữ liệu đầy đủ ở độ phân giải cao hơn, huấn luyện lâu hơn, thử nhiều cách chọn ảnh huấn luyện, tăng số seed, theo dõi xem loss mới có thật sự tác động đến quá trình học hay không, đo đường biên trực tiếp và kiểm tra trên dữ liệu từ nguồn khác. Kết quả hiện tại chỉ mô tả khả năng phân đoạn trên tập dữ liệu đã chọn; chúng không chứng minh độ chính xác chẩn đoán hay giá trị sử dụng trong thực tế lâm sàng.

## Thông tin để chạy lại thí nghiệm

Mã nguồn, kết quả của từng seed, chương trình chuẩn bị dữ liệu, tài liệu tham khảo, biểu đồ và báo cáo kiểm tra trích dẫn đều nằm trong thư mục kết quả. Cả 31 tài liệu tham khảo do pipeline thu thập đã được đối chiếu bằng DOI ở bước cuối. Số liệu gốc cần dùng để kiểm tra là tệp `code/results.json`. Bản tiếng Việt này được viết từ báo cáo tiếng Anh đã sửa theo số liệu thật, không dựa vào các con số sai trong bản thảo tự động ban đầu.
