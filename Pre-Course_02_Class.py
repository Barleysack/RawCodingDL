

class person(object): #상속받는 객체 
    def __init__(self ,name ,position,back_number):
        self.name = name #여기는 속성파트 , 왼쪽에 할당하는 구조. 
        self.position =position
        self.back_number = back_number
    def __str__(self):
        return "hi there!"
a = person("kim",1,21)

print(a.name)