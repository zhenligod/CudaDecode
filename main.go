package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
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

func secp256k1Decode(privkey []byte) (pubkey []byte) {
	pubkey, err := secp256k1.GeneratePubKey(privkey)
	if err != nil {
		panic(err)
	}
	fmt.Println("secp256k1 pubkey:", pubkey)
	return pubkey
}

//New Decode
func DecodeKeyInput(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var keyInput KeyInput
	_ = json.NewDecoder(r.Body).Decode(&keyInput)
	keyInput.ID = strconv.Itoa(R.Intn(100))
	KeyInputs = append(KeyInputs, keyInput)
	seckey, _ := hex.DecodeString(keyInput.Key)
	pubkey := secp256k1Decode(seckey)
	hash := sha256.Sum256(pubkey)
	json.NewEncoder(w).Encode(hash[:])
}

// main the function where execution of the program begins
func main() {
	//Init router
	r := mux.NewRouter()
	r.HandleFunc("/api/KeyInputs", DecodeKeyInput).Methods("POST")
	log.Fatal(http.ListenAndServe(":8000", r))
}
