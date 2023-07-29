package neuro

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"reflect"
	"sync"
	"time"
	//"github.com/jinzhu/copier"
)

type Perc struct {
	Mtx     sync.Mutex `json:"-"`
	Value   float64    `json:"value"`
	PreVals []float64  `json:"pre_vals"`
	Weights []float64  `json:"weights"`
	Error   float64    `json:"error"`
	Start   bool       `json:"start"`
	Final   bool       `json:"final"`
	Bias    bool       `json:"bias"`
}

type DataTeach struct {
	Inputs  []float64 `json:"inputs"`
	Outputs []float64 `json:"outputs"`
	Price   float64   `json:"price"`
	BuySell int       `json:"buy_sell"`
}

type Result struct {
	Percent float64 `json:"accuracy"`
	False   int     `json:"false"`
	True    int     `json:"true"`
}

type NetPerc struct {
	Layers      int         `json:"layer"`
	Neurons     int         `json:"neurons"`
	Inps        int         `json:"inps"`
	Outs        int         `json:"outs"`
	Iters       int         `json:"iters"`
	CurrInd     int         `json:"curr_ind"`
	Error       float64     `json:"error"`
	LearnRate   float64     `json:"learn_rate"`
	LastPrice   float64     `json:"last_price"`
	Result      Result      `json:"result"`
	Bias        bool        `json:"bias"`
	FinalAct    bool        `json:"final_act"`
	Regress     bool        `json:"regress"`
	Budget      float64     `json:"budget"`
	DiffPerce   float64     `json:"diff_perce"`
	StatusBSell bool        `json:"status_buy_sell"`
	ErrorArr    []float64   `json:"error_arr"`
	RandWeights []float64   `json:"random_waights"`
	Data        []DataTeach `json:"data"`
	Net         [][]*Perc   `json:"net"`
	Score       float64     `json:"score"`
	Nols        int         `json:"nols"`
	Trades      int         `json:"trades"`
}

var mtx sync.Mutex

func InitNetPerc(layer, neurons int) *NetPerc {
	n := &NetPerc{
		Layers:  layer,
		Neurons: neurons,
		Result:  Result{},
		Net:     [][]*Perc{},
	}
	n.SetFinAct(true)
	n.SetBias(false)
	return n
}

func (n *NetPerc) SetRegress(reg bool) *NetPerc {
	n.Regress = reg
	return n
}

func (n *NetPerc) SetFinAct(act bool) *NetPerc {
	n.FinalAct = act
	return n
}

func (n *NetPerc) SetBias(bias bool) *NetPerc {
	n.Bias = bias
	return n
}

func (n *NetPerc) LRate(rate float64) *NetPerc {
	n.LearnRate = rate
	return n
}

func (n *NetPerc) CreateNetGraph(data []DataTeach, out int) *NetPerc {
	n.Inps = len(data[0].Inputs)
	n.Outs = out
	n.SetDataAll(data)
	n.SetFinAct(false)
	n.SetBias(false)
	n.SetRegress(true)
	var inps, outs []*Perc
	for i := 0; i < n.Inps; i++ {
		inps = append(inps, &Perc{Start: true})
	}
	if n.Bias {
		inps = append(inps, n.AddBias())
	}
	n.Net = append(n.Net, inps)
	for i := 0; i < n.Layers; i++ {
		perces := []*Perc{}
		for p := 0; p < n.Neurons; p++ {
			perces = append(perces, &Perc{})
		}
		if n.Bias {
			perces = append(perces, n.AddBias())
		}
		n.Net = append(n.Net, perces)
	}
	for i := 0; i < n.Outs; i++ {
		outs = append(outs, &Perc{Final: true})
	}
	n.Net = append(n.Net, outs)
	return n.InitWeight()
}

func (n *NetPerc) CreateNet(data []DataTeach, iteration int) *NetPerc {
	n.Inps = len(data[0].Inputs)
	n.Outs = len(data[0].Outputs)
	n.Iters = iteration
	n.SetDataAll(data)
	var inps, outs []*Perc
	for i := 0; i < n.Inps; i++ {
		inps = append(inps, &Perc{Start: true})
	}
	if n.Bias {
		inps = append(inps, n.AddBias())
	}
	n.Net = append(n.Net, inps)
	for i := 0; i < n.Layers; i++ {
		perces := []*Perc{}
		for p := 0; p < n.Neurons; p++ {
			perces = append(perces, &Perc{})
		}
		if n.Bias {
			perces = append(perces, n.AddBias())
		}
		n.Net = append(n.Net, perces)
	}
	for i := 0; i < n.Outs; i++ {
		outs = append(outs, &Perc{Final: true})
	}
	n.Net = append(n.Net, outs)
	return n.InitWeight()
}

func (n *NetPerc) AddBias() *Perc {
	return &Perc{
		Value: 1,
		Bias:  true,
	}
}

func (n *NetPerc) Operate(rsp []float64, dt DataTeach) {
	if rsp[0] == 1 {
		return
	}
	if rsp[1] == 1 {
		if !n.StatusBSell {
			n.Budget = n.Budget - dt.Price
			n.StatusBSell = true
			n.LastPrice = dt.Price
		}
	}
	if rsp[2] == 1 {
		if n.StatusBSell {
			n.DiffPerce += dt.Price - n.LastPrice
			n.Budget = n.Budget + dt.Price
			n.Trades = n.Trades + 1
			n.StatusBSell = false
		}
	}
}

func (n *NetPerc) SetWeight(min, max float64) *NetPerc {
	n.RandWeights = []float64{min, max}
	return n
}

func (n *NetPerc) InitWeight() *NetPerc {
	if len(n.RandWeights) < 2 {
		n.RandWeights = []float64{-10, 10}
	}
	for i, els := range n.Net {
		for _, el := range els {
			if len(n.Net) != i+1 {
				if n.Bias {
					if len(n.Net)-2 == i {
						el.Weights = n.getWeightsArr(len(n.Net[i+1]))
					} else {
						el.Weights = n.getWeightsArr(len(n.Net[i+1]) - 1)
					}
				} else {
					el.Weights = n.getWeightsArr(len(n.Net[i+1]))
				}
			}
		}
	}
	return n
}

func (n *NetPerc) SetData(data DataTeach) *NetPerc {
	n.Data = append(n.Data, data)
	return n
}

func (n *NetPerc) AppendSetData(data DataTeach) *NetPerc {
	list := []DataTeach{data}
	n.Data = append(list, n.Data...)
	return n
}

func (n *NetPerc) SetDataAll(data []DataTeach) *NetPerc {
	n.Data = append(n.Data, data...)
	return n
}

func (n *NetPerc) SetDataAllNew(data []DataTeach) *NetPerc {
	n.Data = data
	return n
}

func (n *NetPerc) nextData() {
	if len(n.Data) == n.CurrInd+1 {
		n.CurrInd = 0
	} else {
		n.CurrInd += 1
	}
}

func (n *NetPerc) setInputs() {
	for i, el := range n.Net[0] {
		if !el.Bias {
			el.Value = n.getData().Inputs[i]
		}
	}
}

func (n *NetPerc) setInps(data []float64) {
	for i, el := range n.Net[0] {
		if !n.Bias {
			el.Value = data[i]
		} else {
			if i < len(n.Net[0])-1 {
				el.Value = data[i]
			}
		}
	}
}

func (n *NetPerc) Accuracy(min float64) bool {
	return n.Result.Percent >= min
}

func (n *NetPerc) getData() DataTeach {
	return n.Data[n.CurrInd]
}

func (n *NetPerc) logIter(i, iter int) {
	if i%iter == 0 {
		log.Println("iteration: ", i, "; error: ", n.Error)
	}
}

func (n *NetPerc) sigmoid(val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}

func (n *NetPerc) derivative(val float64) float64 {
	return val * (1 - val)
}

var mtxPre sync.Mutex

func (p *Perc) addVals(preVal float64) {
	if p.PreVals == nil {
		p.PreVals = []float64{}
	}
	p.PreVals = append(p.PreVals, preVal)
}

func (p *Perc) sigmoid(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

func (p *Perc) activation() {
	if len(p.PreVals) > 0 {
		var tm float64
		for _, v := range p.PreVals {
			tm += v
		}
		p.Value = p.sigmoid(tm)
		p.PreVals = []float64{}
	}
}

func (p *Perc) activationWithOutAct() {
	if len(p.PreVals) > 0 {
		var tm float64
		for _, v := range p.PreVals {
			tm += v
		}
		p.Value = tm
		p.PreVals = []float64{}
	}
}

func (n *NetPerc) forwardPass() {
	for il, layer := range n.Net {
		for _, perc := range layer {
			if len(n.Net)-1 == il {
				if n.FinalAct {
					perc.activation()
				} else {
					perc.activationWithOutAct()
				}
			} else {
				perc.activation()
			}
			if len(perc.Weights) > 0 {
				for iw, weight := range perc.Weights {
					n.Net[il+1][iw].addVals(perc.Value * weight)
				}
			}
		}
	}
}

func (n *NetPerc) getOut(index int) *Perc {
	return n.Net[len(n.Net)-1][index]
}

func (n *NetPerc) getOuts() []*Perc {
	return n.Net[len(n.Net)-1]
}

func (n *NetPerc) calcErrorIter() {
	var allErr float64
	for i, o := range n.getData().Outputs {
		perc := n.getOut(i)
		perc.Error = o - perc.Value
		allErr += math.Pow(perc.Error, 2)
	}
	n.Error = allErr
}

func (n *NetPerc) calcError() {

	// main error
	var allErr float64
	for i, o := range n.getData().Outputs {
		perc := n.getOut(i)
		perc.Error = o - perc.Value
		allErr += math.Pow(perc.Error, 2)
	}
	n.Error = toFixed(allErr, 10)
	if n.ErrorArr == nil {
		n.ErrorArr = []float64{}
	}
	n.ErrorArr = append(n.ErrorArr, allErr)

	// other perc error
	for il := len(n.Net) - 2; il > 0; il-- {
		for _, perc := range n.Net[il] {
			perc.Error = 0
			for iw, weight := range perc.Weights {
				perc.Error += weight * n.Net[il+1][iw].Error
			}
			perc.Error = perc.Error * perc.proizvod()
		}
	}

}

func (n *NetPerc) backPropogation() {
	for il, layer := range n.Net {
		for _, perc := range layer {
			for iw, weight := range perc.Weights {
				newWeight := weight + n.LearnRate*n.Net[il+1][iw].Error*perc.Value //*n.Net[il+1][iw].proizvod()
				perc.Weights[iw] = newWeight
			}
		}
	}
}

func (p *Perc) proizvod() float64 {
	return p.Value * (1 - p.Value)
}

func (n *NetPerc) Train(showIter ...int) {
	var iter int
	start := time.Now()
	if len(showIter) > 0 {
		iter = showIter[0]
	} else {
		iter = 50
	}
	n.setInputs()
	for i := 0; i < n.Iters; i++ {

		for e := 0; e < len(n.Data); e++ {
			// ===========================
			n.setInputs()
			// ===========================
			n.forwardPass()
			// ===========================
			n.calcError()
			// ===========================
			n.backPropogation()
			// ===========================
			n.nextData()
			// ===========================
		}
		n.calcMainErrorDataSet()
		n.logIter(i, iter)
		n.CurrInd = 0

	}
	n.ErrorArr = []float64{}
	//n.Data = nil
	log.Println("teach time:", time.Now().Sub(start).String())
}

func (n *NetPerc) TrainIter() {
	n.CurrInd = randInt(len(n.Data) - 1)
	n.setInputs()
	// ===========================
	n.forwardPass()
	// ===========================
	n.calcErrorIter()
	//println(fmt.Sprintf("%.20f", n.Error))
	// ===========================
}

func (n *NetPerc) DataCopy() *NetPerc {
	var list []DataTeach
	for i := 0; i < len(n.Data); i++ {
		dt := DataTeach{}
		copy(dt.Inputs, n.Data[i].Inputs)
		copy(dt.Outputs, n.Data[i].Outputs)
		list = append(list, dt)
	}
	n.Data = list
	return n
}

func (n *NetPerc) TrainIters() *NetPerc {
	for e := 0; e < len(n.Data); e++ {
		// ===========================
		n.setInputs()
		// ===========================
		n.forwardPass()
		// ===========================
		n.calcError()
		// ===========================
		n.backPropogation()
		// ===========================
		n.nextData()
		// ===========================
	}
	n.calcMainErrorDataSet()
	return n
}
func (n *NetPerc) TrainItersNew() *NetPerc {
	n.CurrInd = randInt(len(n.Data) - 1)
	n.setInputs()
	n.forwardPass()
	n.calcErrorIter()
	n.calcMainErrorDataSet()
	return n
}

func (n *NetPerc) calcMainErrorDataSet() {
	var sumErr float64 = 0
	for _, fl := range n.ErrorArr {
		if math.IsNaN(fl) {
			sumErr += 1
		} else {
			sumErr += fl
		}
	}
	n.ErrorArr = []float64{}
	n.Error = sumErr / float64(len(n.Data))
}

func (n *NetPerc) PredictClear(data []float64) []float64 {
	n.CurrInd = 0
	n.setInps(data)
	n.forwardPass()
	var response []float64
	for _, perc := range n.getOuts() {
		if n.Regress {
			response = append(response, toFixed(perc.Value, 3))
		} else {
			//response = append(response, roundFl(toFixed(perc.Value, 3)))
			response = append(response, perc.Value)
		}
	}
	return sortedFls(response)
}

func sortedFls(fls []float64) []float64 {
	var max float64
	var maxI int
	for i := len(fls) - 1; i >= 0; i-- {
		if fls[i] > max {
			max = fls[i]
			maxI = i
		}
	}
	flRet := make([]float64, len(fls))
	flRet[maxI] = 1
	return flRet
}

func (n *NetPerc) clearData() {
	n.Data = n.Data[1:]
}

func (n *NetPerc) Predict(data []float64) []float64 {
	n.Data = nil
	n.CurrInd = 0
	n.SetData(DataTeach{Inputs: data})
	n.setInputs()
	n.forwardPass()
	var response []float64
	for _, perc := range n.getOuts() {
		if n.Regress {
			response = append(response, toFixed(perc.Value, 3))
		} else {
			response = append(response, roundFl(perc.Value))
		}
	}
	return response
}

func roundFl(x float64) float64 {
	t := math.Trunc(x)
	if math.Abs(x-t) >= 0.5 {
		return t + math.Copysign(1, x)
	}
	return t
}

func (n *NetPerc) PredictBot(data []float64, last ...int) []float64 {
	n.Data = nil
	n.CurrInd = 0
	n.SetData(DataTeach{Inputs: data})
	n.setInputs()
	n.forwardPass()
	var response []float64
	for _, perc := range n.getOuts() {
		if len(last) > 0 {
			response = append(response, toFixed(perc.Value, last[0]))
		} else {
			response = append(response, toFixed(perc.Value, 3))
		}
	}
	return response
}

func (n *NetPerc) Equal(d1, d2 []float64) bool {
	res := reflect.DeepEqual(d1, d2)
	if res {
		n.Result.True += 1
	} else {
		n.Result.False += 1
	}
	return res
}

func (n *NetPerc) CalcStatAll(data []*DataTeach) *NetPerc {

	n.Result.Percent = 0
	n.Result.True = 0
	n.Result.False = 0

	for i := 0; i < len(data); i++ {
		n.Equal(n.Predict(data[i].Inputs), data[i].Outputs)
	}

	n.Result.Percent = (float64(n.Result.True-n.Result.False) / (float64(n.Result.False+n.Result.True) / 2) * 100) / 2
	return n
}

func (n *NetPerc) CalcStatRegress(data []*DataTeach, count int) *NetPerc {
	for i := 0; i < count; i++ {
		index := randInt(len(data))
		inp := n.Predict(data[index].Inputs)
		out := data[index].Outputs
		for in, fl := range inp {
			inp[in] = toFixed(fl, 3)
		}
		for in, fl := range out {
			out[in] = toFixed(fl, 3)
		}
		n.Equal(inp, out)
	}
	n.Result.Percent = (float64(n.Result.True-n.Result.False) / (float64(n.Result.False+n.Result.True) / 2) * 100) / 2
	return n
}

var mtxCalc sync.Mutex

func (n *NetPerc) CalcStat(count int) *NetPerc {
	mtxCalc.Lock()
	defer mtxCalc.Unlock()
	//n.Result = Result{}
	for i := 0; i < count; i++ {
		index := randInt(len(n.Data) - 1)
		n.Equal(n.PredictClear(n.Data[index].Inputs), n.Data[index].Outputs)
	}
	n.Result.Percent = (float64(n.Result.True-n.Result.False) / (float64(n.Result.False+n.Result.True) / 2) * 100) / 2
	return n
}

func (n *NetPerc) GetStat() string {
	return "accuracy: " + fmt.Sprintf("%.2f", toFixed(n.Result.Percent, 1)) + "%"
}

func (n *NetPerc) Save(fileName string) error {
	if fileName == "" {
		return errors.New("empty filename")
	}
	if bts, err := json.Marshal(n); err != nil {
		return err
	} else {
		if err := ioutil.WriteFile(fileName, bts, 0644); err != nil {
			return err
		}
	}
	return nil
}

func LoadNet(fileName string) (*NetPerc, error) {
	var net NetPerc
	if fileName == "" {
		return &net, errors.New("empty filename")
	}
	bts, err := ioutil.ReadFile(fileName)
	if err != nil {
		return &net, err
	}
	if err := json.Unmarshal(bts, &net); err != nil {
		return &net, err
	}
	return &net, nil
}

func (n *NetPerc) getWeights(val float64, length int) []float64 {
	return randFloats(n.RandWeights[0], n.RandWeights[1], length)
}

func (n *NetPerc) getWeightsArr(length int) []float64 {
	return randFloats(n.RandWeights[0], n.RandWeights[1], length)
}

func randFloats(min, max float64, n int) []float64 {
	res := make([]float64, n)
	for i := range res {
		res[i] = toFixed(min+rand.Float64()*(max-min), 3)
	}
	return res
}

func randFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func randInt(max int) int {
	return rand.Intn(max)
}

func randIntMin(min, max int) int {
	defer func() {
		if msg := recover(); msg != nil {
			fmt.Println(min, max)
			debug(msg)
		}
	}()
	if min > max {
		return min
	}
	if min == max {
		return min
	}
	if max < 0 {
		return 0
	}
	if min == 0 && max == 0 {
		return 0
	}
	if min == 0 && max == 1 {
		return 0
	}
	if max-min == 0 {
		return 0
	}
	return rand.Intn(max-min) + min
}

func getDiff(newPrice, oldPrice float64) float64 {
	return toFixed(((newPrice - oldPrice) / ((newPrice + oldPrice) / 2) * 100), 3)
}

func Diff(newPrice, oldPrice float64) float64 {
	return toFixed(((newPrice - oldPrice) / ((newPrice + oldPrice) / 2) * 100), 3)
}

func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(round(num*output)) / output
}

func round(num float64) int {
	return int(num + math.Copysign(0.5, num))
}

func (n *NetPerc) Copy() *NetPerc {
	var nn NetPerc
	bts, err := json.Marshal(n)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(bts, &nn); err != nil {
		log.Fatal(err)
	}
	nn.Error = 1
	nn.ErrorArr = []float64{}
	nn.Result = Result{}
	return &nn
}

func (n *NetPerc) mutateWeight(min, max float64) {
	layer := randInt(n.Layers + 1)
	per := randInt(len(n.Net[layer]) - 1)
	weightsLength := 0
	if len(n.Net[layer][per].Weights)-1 > 0 {
		weightsLength = randInt(len(n.Net[layer][per].Weights) - 1)
	}
	n.Net[layer][per].Weights[weightsLength] = randFloat(min, max)
}

func debug(in interface{}) {
	bt, err := json.Marshal(in)
	if err != nil {
		log.Fatal("error debug: ", err)
	}
	println(string(bt))
	os.Exit(0)
}
