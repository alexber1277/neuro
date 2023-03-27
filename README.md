```
package main

import (
	"encoding/json"
	"log"

	"github.com/alexber1277/neuro"
)

const (
	fileNameDump = "basenet.data"
)

func main() {

	inpData := []*neuro.DataTeach{
		&neuro.DataTeach{[]float64{1, 2, 3}, []float64{1}},
		&neuro.DataTeach{[]float64{2, 3, 4}, []float64{0}},
		&neuro.DataTeach{[]float64{3, 4, 5}, []float64{1}},
		&neuro.DataTeach{[]float64{4, 5, 6}, []float64{0}},
		&neuro.DataTeach{[]float64{5, 6, 7}, []float64{1}},
		&neuro.DataTeach{[]float64{6, 7, 8}, []float64{0}},
		&neuro.DataTeach{[]float64{7, 8, 9}, []float64{1}},
		&neuro.DataTeach{[]float64{8, 9, 10}, []float64{0}},
		&neuro.DataTeach{[]float64{9, 10, 11}, []float64{1}},
	}

	net, err := neuro.LoadNet(fileNameDump) // load net by filename

	if err != nil {
		log.Println(err)
		net = neuro.InitNetPerc(2, 120) // initialisation (2 layers by 120 neurons)
		net.LRate(0.01)                 // learning rate
		net.CreateNet(inpData, 5000)    // set teach data and epoch
		net.Train(100)                  // show result by (n) iteration
	}

	net.CalcStat(inpData, 100).GetStat() // check result by random teach data and get statistics
	debug(net.Result)                    // show result  {"accuracy":100,"false":0,"true":100}

	if net.Accuracy(99.0) { // set minimal right answers percent
		//net.Save(fileNameDump) // save dump
	}

}

func debug(in interface{}) {
	bt, err := json.Marshal(in)
	if err != nil {
		log.Fatal("error debug: ", err)
	}
	log.Println(string(bt))
}
```