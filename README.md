## Neural network
### - Training with back propagation

```golang
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"reflect"
	"runtime"
	"time"

	"github.com/alexber1277/neuro"
)

const (
	fileNameDump = "basenet.data"
)

var inpData = []neuro.DataTeach{
	neuro.DataTeach{[]float64{1, 2, 3}, []float64{1}},
	neuro.DataTeach{[]float64{2, 3, 4}, []float64{0}},
	neuro.DataTeach{[]float64{3, 4, 5}, []float64{1}},
	neuro.DataTeach{[]float64{4, 5, 6}, []float64{0}},
	neuro.DataTeach{[]float64{5, 6, 7}, []float64{1}},
}

func init() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
	net, err := neuro.LoadNet(fileNameDump) // load net by filename

	if err != nil {
		log.Println(err)
		net = neuro.InitNetPerc(2, 120) // initialisation (2 layers by 120 neurons)
		net.LRate(0.01)                 // learning rate
		net.CreateNet(inpData, 1000)    // set teach data and epoch
		net.Train(100)                  // show result by (n) iteration
	}

	net.CalcStat(100).GetStat() // check result by random teach data and get statistics

	debug(net.Result) // show result  {"accuracy":100,"false":0,"true":100}
	debug(net.Error)  // result error percent
	debug(net)        // result error percent

	if net.Accuracy(99.0) { // set minimal right answers percent
		net.Save(fileNameDump) // save dump
	}
}

func debug(in interface{}) {
	bt, err := json.Marshal(in)
	if err != nil {
		log.Fatal("error debug: ", err)
	}
	println(string(bt))
}

```


### - Training with genetic algorithm
```golang
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"reflect"
	"runtime"
	"time"

	"github.com/alexber1277/neuro"
)

const (
	fileNameDump = "basenet.data"
)

var inpData = []neuro.DataTeach{
	neuro.DataTeach{[]float64{1, 2, 3}, []float64{1}},
	neuro.DataTeach{[]float64{2, 3, 4}, []float64{0}},
	neuro.DataTeach{[]float64{3, 4, 5}, []float64{1}},
	neuro.DataTeach{[]float64{4, 5, 6}, []float64{0}},
	neuro.DataTeach{[]float64{5, 6, 7}, []float64{1}},
}

func init() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {

	gen := neuro.InitGenetic()
	gen.Add(func() *neuro.NetPerc {
		return neuro.InitNetPerc(2, 60).
			SetWeight(gen.Config.MinRandWeight, gen.Config.MaxRandWeight).
			LRate(0.1).
			CreateNet(inpData, 1)
	})

	gen.Error = 1
	for i := 0; gen.Error > gen.Config.BestResult; i++ {
		gen.Train(false)
		log.Println("iter:", i, "best result percent:", gen.Error)
	}

	n := gen.GetBest()

	for _, el := range inpData {
		fmt.Println(
			el.Inputs, "\t",
			n.PredictClear(el.Inputs), "\t",
			el.Outputs, "\t",
			reflect.DeepEqual(n.PredictClear(el.Inputs), el.Outputs),
		)
	}
}

func debug(in interface{}) {
	bt, err := json.Marshal(in)
	if err != nil {
		log.Fatal("error debug: ", err)
	}
	println(string(bt))
}

```