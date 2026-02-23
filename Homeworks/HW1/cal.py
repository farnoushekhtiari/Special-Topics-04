while True:
    try:
        a = float(input("عدد اول: "))
        op = input("عملگر (+ - * /): ")
        b = float(input("عدد دوم: "))
        
        if op == '+':
            print(a + b)
        elif op == '-':
            print(a - b)
        elif op == '*':
            print(a * b)
        elif op == '/':
            if b == 0:
                print("خطا: تقسیم بر صفر")
            else:
                print(a / b)
        else:
            print("عملگر اشتباه")
            
    except ValueError:
        print("لطفاً عدد درست وارد کن")
    except Exception as e:
        print("خطا:", e)