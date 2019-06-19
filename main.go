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

	"github.com/gorilla/mux"
	"github.com/zhenligod/go-ethereum/crypto/secp256k1"
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

//secp256k1 handle function
func DecodeSecp256k1Input(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var keyInput KeyInput
	_ = json.NewDecoder(r.Body).Decode(&keyInput)
	keyInput.ID = strconv.Itoa(R.Intn(100))
	KeyInputs = append(KeyInputs, keyInput)
	seckey, _ := hex.DecodeString(keyInput.Key)
	pubkey := secp256k1Decode(seckey)
	json.NewEncoder(w).Encode(pubkey)
}

//sha256 handle function
func DecodeSha256Input(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var keyInput KeyInput
	_ = json.NewDecoder(r.Body).Decode(&keyInput)
	keyInput.ID = strconv.Itoa(R.Intn(100))
	KeyInputs = append(KeyInputs, keyInput)
	seckey, _ := hex.DecodeString(keyInput.Key)
	res := sha256.Sum256(seckey)
	json.NewEncoder(w).Encode(res)
}

// main the function where execution of the program begins
func main() {
	//Init router
	r := mux.NewRouter()
	r.HandleFunc("/api/secp256k1", DecodeSecp256k1Input).Methods("POST")
	r.HandleFunc("/api/sha256", DecodeSha256Input).Methods("POST")
	log.Fatal(http.ListenAndServe(":8000", r))
}
