import usb.core
# Bus 001 Device 041: ID 04d8:fe97 Microchip Technology, Inc. 

class SmartPalate:
	def __init__(self):
		self.USB_BUFFER_SIZE = 512
		self.SENSOR_COUNT = 124
		self.SENSOR_SET_COUNT = 17
		self.SENSOR_DATA_SET_COUNT = 2 * self.SENSOR_SET_COUNT

		self.device = self._getUSBDevice()
		if self.device is None:
			raise Exception('Smart Palate not found')
		self.last_address = 0
		self.palate_frame = []
		self.palate_byte_count = 0
		
		self.palate_set = [0 for i in range(self.SENSOR_DATA_SET_COUNT)]
		for b in range(1, self.SENSOR_SET_COUNT + 1):
			self.palate_set[b * 2 - 1] = b

	def _getUSBDevice(self):
		smart_palate = usb.core.find(idVendor=0x04d8, idProduct=0xfe97)
		if smart_palate != None:
			print('smart palate found!')
			return smart_palate

		print('no smart palate found')
		return None

	def readFrame(self):
		data = None

		if self.device != None:
			try:
				data = self.device.read(0x81, 512, 1)
			except:
				print(sys.exc_info()[0])
				raise

		if data is None or len(data) != self.USB_BUFFER_SIZE:
			raise Exception('Missing data!')
			return None

		return self._parsePacket(data)

	def _parsePacket(self, packet):
		palate_buffer = None
		if len(packet) > 4:
			read_buffer = packet
			source_index = 6
			b = read_buffer[2]
			array = [0 for i in range(100)]
			array[3 : 3 + b*2] = read_buffer[source_index : source_index + b*2]
			array[0] = read_buffer[0]
			array[1] = read_buffer[1]
			array[2] = b
			self.palate_frame.append(array)
			self.palate_byte_count += b * 2
			# print(self.palate_byte_count, self.SENSOR_DATA_SET_COUNT)

			if self.palate_byte_count >= self.SENSOR_DATA_SET_COUNT:
				palate_buffer = self._onNewPalateData(self.palate_frame)[0]
				self.palate_frame = []
				self.palate_byte_count = 0

		return palate_buffer

	def _onNewPalateData(self, new_frame):
		output = []
		count_history = []

		for current in new_frame:
			b6 = current[0]
			b7 = current[1]
			b = current[2]
			packet_count = b6 + (b7 << 8)
			count_history.append(packet_count)

			flag = False
			for j in range(3, 2 + b * 2 +1, 2):
				b3 = current[j]
				b4 = current[j + 1]

				if b4 > 17:
					pass
				else:
					self.palate_set[b4 * 2 - 2] = b3
					if self.last_address > b4:
						flag = True
					self.last_address = b4

			if flag:
				output.append((count_history, self._redrawSensors(self.palate_set)))
				count_history = []

		return output

	def _redrawSensors(self, new_frame):
		num = 0
		output = [0 for i in range(self.SENSOR_COUNT)]

		while True:
			# not sure why but this doesn't work...
			# if num >= 17:
				# print 'bye'
				# return

			if num * 2 + 1 > len(new_frame):
				break
			b = new_frame[num * 2]
			b2 = new_frame[num * 2 + 1]
			for i in range(8):
				num2 = (b2 - 1) * 8 + i
				if b2 <= 0 or num2 >= self.SENSOR_COUNT:
					break
				if ((b << 7 - i) >> 7) & 1 > 0:
					output[num2] = 1
				else:
					output[num2] = 0
			num += 1

		return output
