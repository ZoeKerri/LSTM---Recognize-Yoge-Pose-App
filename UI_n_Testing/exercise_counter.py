import time

class ExerciseCounter:
    def __init__(self):
        self.exercise_name = "None"  # Bài tập hiện tại
        # Đếm theo giây
        self.exercise_counter = {
            "Garland_Pose": 0,
            "Happy_Baby_Pose": 0,
            "Head_To_Knee_Pose": 0,
            "Lunge_Pose": 0,
            "Mountain_Pose": 0,
            "Plank_Pose": 0,
            "Raised_Arms_Pose": 0,
            "Seated_Forward_Bend": 0,
            "Staff_Pose": 0,
            "Standing_Forward_Bend": 0
        }
        # Lưu lại thời gian bắt đầu của bài tập để tiến hành cộng thời gian vào counter
        self.start_times = {
            "Garland_Pose": None,
            "Happy_Baby_Pose": None,
            "Head_To_Knee_Pose": None,
            "Lunge_Pose": None,
            "Mountain_Pose": None,
            "Plank_Pose": None,
            "Raised_Arms_Pose": None,
            "Seated_Forward_Bend": None,
            "Staff_Pose": None,
            "Standing_Forward_Bend": None
        }
        self.trust_point = {  # Giá trị giúp chuyển bài tập khi nhận ra bài tập khác liên tục
            "Garland_Pose": 0,
            "Happy_Baby_Pose": 0,
            "Head_To_Knee_Pose": 0,
            "Lunge_Pose": 0,
            "Mountain_Pose": 0,
            "Plank_Pose": 0,
            "Raised_Arms_Pose": 0,
            "Seated_Forward_Bend": 0,
            "Staff_Pose": 0,
            "Standing_Forward_Bend": 0
        }
        self.notification = ""
        self.notification_point = 10  # Ngưỡng frame tiến hành thông báo nhận diện bài tập khác
        self.maximum_point = 20  # Ngưỡng chuyển bài tập khác khi mà trustpoint > maxpoint này

    def reset_trust_point(self):
        for key in self.trust_point:
            self.trust_point[key] = 0
        self.notification = ""
        return self.notification

    def switching_new_exercise(self, new_exercise):
        if self.exercise_name != new_exercise:
            # Kiểm tra xem có bài tập nào đạt ngưỡng maximum_point để chuyển đổi
            if any(value >= self.maximum_point for value in self.trust_point.values()):
                self.reset_trust_point() # Reset tất cả trust points
                self.exercise_name = new_exercise # Cập nhật bài tập hiện tại
                
                # Reset thời gian bắt đầu cho tất cả các bài tập khác
                for exercise in self.start_times:
                    if exercise != new_exercise:
                        self.start_times[exercise] = None
                # Bắt đầu đếm giây cho bài tập mới nếu chưa có
                if self.start_times[new_exercise] is None:
                    self.start_times[new_exercise] = time.time()

                self.notification = "" # Xóa thông báo khi đã chuyển bài tập
                return (self.notification, self.exercise_name)
            
            # Kiểm tra nếu đã đạt ngưỡng notification_point để đưa ra thông báo
            elif any(value >= self.notification_point for value in self.trust_point.values()):
                # Tìm bài tập có trust_point cao nhất để gợi ý chuyển đổi
                potential_new_exercise = max(self.trust_point, key=self.trust_point.get)
                self.notification = f"New exercise detected: {potential_new_exercise}. Switching in {self.maximum_point - self.trust_point[potential_new_exercise]} frames if consistent..."
            else:
                self_notification = "" # Xóa thông báo nếu chưa đủ ngưỡng notification_point

            # Tăng trust point cho bài tập được nhận diện liên tục
            # Đảm bảo chỉ tăng trust_point cho bài tập đang được nhận diện liên tục
            for exercise in self.trust_point.keys():
                if exercise == new_exercise:
                    self.trust_point[exercise] += 1
                else:
                    # Giảm trust_point của các bài tập khác để tránh nhiễu
                    if self.trust_point[exercise] > 0:
                        self.trust_point[exercise] -= 1
        else:
            # Nếu vẫn là bài tập đang thực hiện, reset trust points của các bài tập khác
            self.notification = ""
            for exercise in self.trust_point.keys():
                if exercise != self.exercise_name and self.trust_point[exercise] > 0:
                    self.trust_point[exercise] -= 1
        
        return (self.notification, self.exercise_name)


    def counting(self, predicted_exercise, landmarks):
        # Nếu bài tập nhận diện khác thì thực hiện hàm này để tiến hành đổi bài tập khác hoặc tăng trust point
        if predicted_exercise != self.exercise_name:
            return self.switching_new_exercise(predicted_exercise)

        # Nếu không có bài tập đang được theo dõi (exercise_name == "None") thì khởi tạo
        if self.exercise_name == "None":
            # Nếu bài tập được dự đoán không phải "None" thì gán và bắt đầu đếm
            if predicted_exercise != "None":
                self.exercise_name = predicted_exercise
                self.start_times[self.exercise_name] = time.time()
                self.reset_trust_point() # Reset trust points khi bắt đầu bài tập mới
                return (self.notification, self.exercise_name)
            else: # Nếu predicted_exercise cũng là "None" thì không làm gì cả
                return ("", "None")


        # Nếu vẫn là bài tập đang đếm, cập nhật thời gian
        # Khởi tạo bộ đếm nếu chưa có (trường hợp mới chuyển đổi hoặc khởi động)
        if self.start_times[self.exercise_name] is None:
            self.start_times[self.exercise_name] = time.time()

        # Tăng bộ đếm khi đúng bài tập
        elapsed_time = int(time.time() - self.start_times[self.exercise_name])
        self.exercise_counter[self.exercise_name] = elapsed_time
        
        # Đảm bảo các trust_point của các bài tập khác được giảm khi bài tập hiện tại được xác nhận
        self.notification = ""
        for exercise in self.trust_point.keys():
            if exercise != self.exercise_name and self.trust_point[exercise] > 0:
                self.trust_point[exercise] -= 1

        return (self.notification, self.exercise_name)