package socket

import (
	"fmt"
	"net"
	"os"
)

func sender(conn net.Conn, words string) {
	conn.Write([]byte(words))
	fmt.Println("send over")
}

func SendPacketClient(serverAddr string, words string) (error, string) {
	tcpAddr, err := net.ResolveTCPAddr("tcp4", serverAddr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		return err, ""
	}

	conn, err := net.DialTCP("tcp", nil, tcpAddr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		return err, ""
	}

	fmt.Println("connect success")
	sender(conn, words)
	buffer := make([]byte, 2048)
	n, err := conn.Read(buffer)
	fmt.Println(string(buffer[:n]))
	return nil, string(buffer[:n])
}
