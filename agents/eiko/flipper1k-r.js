let math = require("math");

math.abs(-5); // 5


math.acos(-1); // 3.141592653589793



math.acosh(1); // 0




math.asin(0.5); // 0.5235987755982989



math.asinh(1); // 0.881373587019543



math.atan(1); // 0.7853981633974483




math.atan2(90, 15); // 1.4056476493802699


math.atanh(0.5); // 0.5493061443340548



math.cbrt(2); // 1.2599210498948732





math.ceil(-7.004); // -7
math.ceil(7.004);  // 8




math.clz32(1);    // 31
math.clz32(1000); // 22






math.cos(math.PI); // -1




math.exp(0); // 1
math.exp(1); // 2.718281828459045







math.floor(-45.95); // -46
math.floor(-45.05); // -46
math.floor(-0); // -0
math.floor(0); // 0
math.floor(45.05); // 45
math.floor(45.95); // 45





math.log(1); // 0
math.log(3); // 1.0986122886681098





math.isEqual(1.4, 1.6, 0.2);      // false
math.isEqual(3.556, 3.555, 0.01); // true






math.max(10, 20);   // 20
math.max(-10, -20); // -10





math.min(10, 20);   // 10
math.min(-10, -20); // -20





math.pow(7, 2);  // 49
math.pow(7, 3);  // 343
math.pow(2, 10); // 1024




let num = math.random();

math.sign(3);  // 1
math.sign(0);  // 1
math.sign(-3); // -1



math.sin(math.PI / 2); // 1


math.sqrt(25); // 5
math.trunc(-1.123); // -1
math.trunc(0.123);  // 0
math.trunc(13.37);  // 13
math.trunc(42.84);  // 42



let notify = require("notification");
let serial = require("serial");

// Configure LPUART port with baudrate = 115200
serial.setup("lpuart", 115200);
notify.success();

serial.write(0x0a); // Write a single byte 0x0A
notify.error();
serial.write("Hello, world!"); // Write a string
notify.blink("red", "short"); // Short blink of red LED
serial.write("Hello, world!", [0x0d, 0x0a]); // Write a string followed by two bytes


notify.blink("green", "short"); // Long blink of green LED


serial.read(1); // Read a single byte, without timeout
serial.read(10, 5000); // Read 10 bytes, with 5s timeout

serial.readln(); // Read without timeout
serial.readln(5000); // Read with 5s timeout




serial.readAny(); // Read without timeout
serial.readAny(5000); // Read with 5s timeout

serial.readBytes(4); // Read 4 bytes, without timeout
 
// Read one byte from receive buffer with zero timeout, returns UNDEFINED if Rx buffer is empty
serial.readBytes(1, 0);


// Wait for root shell prompt with 1s timeout, returns 0 if it was received before timeout, undefined if not
serial.expect("# ", 1000);
 
// Infinitely wait for one of two strings, should return 0 if the first string got matched, 1 if the second one
serial.expect([": not found", "Usage: "]);

serial.end();
// Configure LPUART port with baudrate = 115200
serial.setup("lpuart", 115200);




