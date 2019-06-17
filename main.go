package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/json"
	"log"
	R "math/rand"
	"net/http"
	"strconv"

	"github.com/ethereum/go-ethereum/crypto/secp256k1"
	"github.com/gorilla/mux"
)

var KeyInputs []KeyInput

//KeyInput struct
type KeyInput struct {
	ID  string `json:"id"`
	Key string `json:"Key"`
}

func generateKeyPair() (pubkey, privkey []byte) {
	key, err := ecdsa.GenerateKey(secp256k1.S256(), rand.Reader)
	if err != nil {
		panic(err)
	}
	pubkey = elliptic.Marshal(secp256k1.S256(), key.X, key.Y)

	privkey = make([]byte, 32)
	blob := key.D.Bytes()
	copy(privkey[32-len(blob):], blob)

	return pubkey, privkey
}

//New Decode
func DecodeKeyInput(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var keyInput KeyInput
	_ = json.NewDecoder(r.Body).Decode(&keyInput)
	keyInput.ID = strconv.Itoa(R.Intn(100))
	KeyInputs = append(KeyInputs, keyInput)
	json.NewEncoder(w).Encode(keyInput)
}

// main the function where execution of the program begins
func main() {
	//Init router
	r := mux.NewRouter()
	r.HandleFunc("/api/KeyInputs", DecodeKeyInput).Methods("POST")
	log.Fatal(http.ListenAndServe(":8000", r))
}
