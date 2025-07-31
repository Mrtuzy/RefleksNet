import serial
import time


class ArduinoController:
    def __init__(self, port='COM3', baudrate=9600):
        """
        Arduino ile bağlantı kurar
        Windows: 'COM3', 'COM4' vs.
        Linux/Mac: '/dev/ttyUSB0', '/dev/ttyACM0' vs.
        """
        try:
            self.arduino = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Arduino'nun başlatılması için bekle
            print(f"Arduino ile bağlantı kuruldu: {port}")
        except serial.SerialException as e:
            print(f"Bağlantı hatası: {e}")
            self.arduino = None

    def send_command(self, command):
        """Arduino'ya komut gönder"""
        if self.arduino and self.arduino.is_open:
            self.arduino.write((command + '\n').encode())
            time.sleep(0.1)  # Komutun işlenmesi için bekle

            # Arduino'dan gelen yanıtı oku
            if self.arduino.in_waiting > 0:
                response = self.arduino.readline().decode().strip()
                return response
        return None

    def led_on(self):
        """LED'i aç"""
        response = self.send_command("LED_ON")
        print(f"Arduino yanıtı: {response}")

    def led_off(self):
        """LED'i kapat"""
        response = self.send_command("LED_OFF")
        print(f"Arduino yanıtı: {response}")

    def get_status(self):
        """LED durumunu öğren"""
        response = self.send_command("STATUS")
        print(f"Arduino yanıtı: {response}")
        return response

    def close(self):
        """Bağlantıyı kapat"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Bağlantı kapatıldı")


# Kullanım örneği
def main():
    # Port'u kendi sisteminize göre değiştirin
    controller = ArduinoController('COM3')  # Windows için
    # controller = ArduinoController('/dev/ttyACM0')  # Linux için

    if controller.arduino:
        try:
            while True:
                print("\n--- Arduino Kontrol ---")
                print("1. LED Aç")
                print("2. LED Kapat")
                print("3. Durum Öğren")
                print("4. Çıkış")

                choice = input("Seçiminiz (1-4): ")

                if choice == '1':
                    controller.led_on()
                elif choice == '2':
                    controller.led_off()
                elif choice == '3':
                    controller.get_status()
                elif choice == '4':
                    break
                else:
                    print("Geçersiz seçim!")

        except KeyboardInterrupt:
            print("\nProgram sonlandırılıyor...")
        finally:
            controller.close()


if __name__ == "__main__":
    main()