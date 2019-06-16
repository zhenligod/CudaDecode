package main

import (
	"encoding/json"
	"log"
	"math/rand"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
)

var KeyInputs []KeyInput

//KeyInput struct
type KeyInput struct {
	ID  string `json:"id"`
	Key string `json:"Key"`
}

//New Decode
func DecodeKeyInput(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var keyInput KeyInput
	_ = json.NewDecoder(r.Body).Decode(&keyInput)
	keyInput.ID = strconv.Itoa(rand.Intn(100))
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
