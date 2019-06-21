package socket

import (
	"testing"
)

func TestSocketSha256(t *testing.T) {
	serverAddr := "gzhl-feed-qatest125.gzhl.baidu.com:8000"
	words := "hello world"
	err := sendPacketClient(serverAddr, words)
	if err != nil {
		t.Fatal(err)
	}
}
