package socket

import (
	"testing"
)

func TestSocketSha256(t *testing.T) {
	serverAddr := "localhost:8000"
	words := "hello world"
	err, _ := SendPacketClient(serverAddr, words)
	if err != nil {
		t.Fatal(err)
	}
}
