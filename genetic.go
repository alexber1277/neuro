package neuro

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/thoas/go-funk"
)

type Genetic struct {
	Nets      []*NetPerc  `json:"nets"`
	ResOrders []*ResOrder `json:"res_orders"`
	Config    GeneticConf `json:"conf"`
	Percent   float64     `json:"percent"`
	Error     float64     `json:"error"`
	Score     float64     `json:"score"`
	Iters     int         `json:"iters"`
	LBOitem   *LBO        `json:"lbo_item"`
	Tm        time.Time   `json:"tm"`
}

type GeneticConf struct {
	Population     int          `json:"population"`
	LastBest       int          `json:"last_best"`
	LimitMutateSub int          `json:"limit_mutate_sub"`
	Inps           int          `json:"inps"`
	NewItems       int          `json:"new_items"`
	PercByHours    int          `json:"perc_by_hours"`
	DiffShift      int          `json:"diff_shift"`
	MaxMutateIter  int          `json:"max_mutate_iter"`
	MinRandWeight  float64      `json:"min_weight"`
	MaxRandWeight  float64      `json:"max_weight"`
	BestResult     float64      `json:"best_result"`
	Budget         float64      `json:"budget"`
	TradesByDay    float64      `json:"trades_by_day"`
	Hours          float64      `json:"hours"`
	MinPerce       float64      `json:"min_perce"`
	Data           []*DataTeach `json:"data"`
	DataList       []DataTeach  `json:"data_list"`
}

type LBO struct {
	Score  float64 `json:"score"`
	Count  int     `json:"count"`
	Trades []int   `json:"trades"`
}

var defaultConf = GeneticConf{
	Population:     500,
	LastBest:       100,
	MinRandWeight:  -100,
	MaxRandWeight:  100,
	LimitMutateSub: 100,
	BestResult:     0.001,
	PercByHours:    10,
	Budget:         1000,
	DiffShift:      2,
}

type ResOrder struct {
	Count  int     `json:"count"`
	By     int     `json:"by"`
	Type   bool    `json:"type"`
	OldSum float64 `json:"oldsum"`
	Sum    float64 `json:"sum"`
	Score  float64 `json:"score"`
	Diff   float64 `json:"diff"`
	Trades []int   `json:"trades"`
}

func init() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func InitGenetic(conf ...GeneticConf) *Genetic {
	g := &Genetic{Config: defaultConf}
	g.Tm = time.Now()
	if len(conf) != 0 {
		g.Config = conf[0]
	}
	g.Error = 1
	return g
}

func (g *Genetic) GenerateTrades() *Genetic {
	for i := 0; i < g.Config.Population; i++ {
		g.ResOrders = append(g.ResOrders, &ResOrder{Trades: []int{}})
	}
	return g
}

func (g *Genetic) GenerateTradesV2() *Genetic {
	allCount := int(g.Config.Inps / 100 * g.Config.PercByHours)
	diffCount := int(g.Config.Inps / 100 * g.Config.DiffShift)
	min := allCount - diffCount
	max := allCount + diffCount
	for i := min; i <= max; i++ {
		if i == 0 {
			continue
		}
		if i%2 == 0 {
			g.ResOrders = append(g.ResOrders, &ResOrder{Count: i, Trades: []int{}})
		}
	}
	return g
}

func (g *Genetic) SetWorkOrders(ord *ResOrder) *Genetic {
	g.Iters = 0
	g.Score = 0
	g.LBOitem = &LBO{}
	g.ResOrders = []*ResOrder{ord}
	return g
}

func (g *Genetic) GenerateOrdersV2() *Genetic {
	for _, r := range g.ResOrders {
		r.By = int(g.Config.Inps / r.Count)
		for t := 0; t < g.Config.Inps; t++ {
			if t == 0 {
				continue
			}
			if t%r.By == 0 {
				r.Trades = append(r.Trades, t)
			}
		}
	}
	return g
}

func (g *Genetic) FirstMutate(inpData []DataTeach, gsfd *bool) *Genetic {
	g.LBOitem = &LBO{Trades: []int{}}
	for {
		var wg sync.WaitGroup
		wg.Add(len(g.ResOrders))
		for _, ord := range g.ResOrders {
			go func(ordItem *ResOrder) {
				defer wg.Done()
				ordItem.Sum = g.Config.Budget
				ordItem.Type = false
				//lastPrice := 0.0
				for _, pr := range ordItem.Trades {
					if !ordItem.Type {
						ordItem.Sum -= inpData[pr].Price
						ordItem.Type = true
						//lastPrice = inpData[pr].Price
					} else {
						//if getDiff(inpData[pr].Price, lastPrice) > g.Config.MinPerce {
						ordItem.Sum += inpData[pr].Price
						ordItem.Type = false
						//}
					}
				}
			}(ord)
		}
		wg.Wait()
		g.sortBestOrders()
		if g.Score < g.GetBestOrders().Score {
			g.LBOitem.Count = g.GetBestOrders().Count
			g.LBOitem.Score = g.GetBestOrders().Score
			g.LBOitem.Trades = g.GetBestOrders().Trades
			g.Score = g.GetBestOrders().Score
			g.LogScoreOrders(1, " !!! BEST !!! ")
		} else {
			g.LogScoreOrders(10000, " !!! TIME !!! ")
		}
		if *gsfd {
			break
		}

		g.mutateOrdersFirst()
		g.Iters += 1

		//time.Sleep(time.Second / 10)

	}

	return g
}

func (g *Genetic) GenerateOrders(maxLength int) *Genetic {
	for _, r := range g.ResOrders {
		r.Count = randIntMin(1, int(maxLength/2))
		r.Trades = []int{}
		r.By = int(math.Floor(float64(maxLength / r.Count)))
		for i := 0; i < maxLength-1; i++ {
			if i%r.By == 0 {
				r.Trades = append(r.Trades, i)
			}
		}
		r.Trades = r.Trades[:r.Count]
	}
	return g
}

func (g *Genetic) AddOrder() *ResOrder {
	r := ResOrder{}
	r.Count = randIntMin(1, int(g.Config.Inps))
	r.Trades = []int{}
	r.By = int(math.Floor(float64(g.Config.Inps / r.Count)))
	for i := 0; i < g.Config.Inps-1; i++ {
		if i%r.By == 0 {
			r.Trades = append(r.Trades, i)
		}
	}
	r.Trades = r.Trades[:r.Count]
	return &r
}

func (g *Genetic) GetTm() string {
	return time.Now().Sub(g.Tm).String()
}

func (g *Genetic) GetBest() *NetPerc {
	return g.Nets[0]
}

func (g *Genetic) GetBestOrders() *ResOrder {
	return g.ResOrders[0]
}

func (g *Genetic) GetLBO() []int {
	return g.LBOitem.Trades
}

func (g *Genetic) Add(ret func() *NetPerc) {
	for i := 0; i < g.Config.Population; i++ {
		g.AddNet(ret())
	}
}

func (g *Genetic) SetData(data []*DataTeach) *Genetic {
	g.Config.Data = data
	return g
}

func (g *Genetic) AddNet(net *NetPerc) *Genetic {
	net.Budget = g.Config.Budget
	g.Nets = append(g.Nets, net)
	return g
}

func (g *Genetic) ClearScore() {
	g.Score = 0
}

func (g *Genetic) sortBest() *Genetic {
	sort.Slice(g.Nets, func(i, j int) bool {
		g.Nets[i].Score = g.Nets[i].DiffPerce / (g.Config.Hours/24*g.Config.TradesByDay + float64(g.Nets[i].Trades))
		g.Nets[j].Score = g.Nets[j].DiffPerce / (g.Config.Hours/24*g.Config.TradesByDay + float64(g.Nets[j].Trades))
		return g.Nets[i].Score > g.Nets[j].Score
	})
	return g
}

func (g *Genetic) sortBestOrders() *Genetic {
	sort.Slice(g.ResOrders, func(i, j int) bool {
		g.ResOrders[i].Score = g.ResOrders[i].Sum
		g.ResOrders[j].Score = g.ResOrders[j].Sum
		return g.ResOrders[i].Score > g.ResOrders[j].Score
	})
	return g
}

func (g *Genetic) GetMediana(cnt int) float64 {
	med := (float64(len(g.ResOrders)/2-cnt) / (float64(g.Config.Inps) / 0.00001))
	if med < 0 {
		med *= -1
		return med
	}
	if med == 0 {
		med = 0.00001
		return med
	}
	return med
}

func (g *Genetic) SortResOrdersV2() *Genetic {
	sort.Slice(g.ResOrders, func(i, j int) bool {
		return g.ResOrders[i].Score > g.ResOrders[j].Score
	})
	return g
}

func (g *Genetic) SortResOrders() *Genetic {
	sort.Slice(g.ResOrders, func(i, j int) bool {
		g.ResOrders[i].Score = g.ResOrders[i].Sum / 1000000 / g.GetMediana(g.ResOrders[i].Count)
		g.ResOrders[j].Score = g.ResOrders[j].Sum / 1000000 / g.GetMediana(g.ResOrders[j].Count)
		return g.ResOrders[i].Score > g.ResOrders[j].Score
	})
	return g
}

// perc - percent by inps
func (g *Genetic) GenerateOrdersV3(perc float64) *Genetic {
	start := int(float64(g.Config.Inps) * (perc / 100))
	length := g.Config.Inps - start*2
	for l := length; l >= start; l-- {
		r := ResOrder{}
		r.Trades = []int{}
		for i := start; i < l; i++ {
			if i%3 == 0 {
				r.Trades = append(r.Trades, i)
			}
		}
		r.Count = len(r.Trades)
		if r.Count%2 == 0 && r.Count != 0 {
			g.ResOrders = append(g.ResOrders, &r)
		}
	}
	return g
}

// perc - percent by inps
func (g *Genetic) GenerateOrdersV4(perc int) *Genetic {
	step := int(perc / 3)
	max := perc + step
	min := perc - step
	for i := min; i <= max; i++ {
		var r ResOrder
		for s := 0; s < g.Config.Inps; s++ {
			if s%i == 0 {
				r.Trades = append(r.Trades, s)
			}
		}
		r.Count = len(r.Trades)
		g.ResOrders = append(g.ResOrders, &r)
	}
	return g
}

// perc - percent by inps
func (g *Genetic) GenerateOrdersV5(perc ...int) *Genetic {
	var perceOrig int = 10
	var countOrders int
	if len(perc) > 0 {
		perceOrig = perc[0]
	}
	countOrders = int(float64(g.Config.Inps) * float64(float64(perceOrig)/100))
	if countOrders%2 != 0 {
		countOrders += 1
	}
	r := ResOrder{Count: countOrders}
	mpDt := map[int]int{}
	for {
		in := randIntMin(10, g.Config.Inps)
		if _, ok := mpDt[in]; !ok {
			r.Trades = append(r.Trades, in)
			mpDt[in] = 1
		}
		if len(r.Trades) >= r.Count {
			break
		}
	}
	sort.Ints(r.Trades)
	g.ResOrders = append(g.ResOrders, &r)
	return g
}

func (g *Genetic) AddPopulations() *Genetic {
	var s int
	for i := 0; i < g.Config.Population; i++ {
		next := g.ResOrders[s].CopyJson()
		g.ResOrders = append(g.ResOrders, next.mutateV3(g.Config.Inps))
		s += 1
	}
	return g
}

func (g *Genetic) sortBestNols() *Genetic {
	sort.Slice(g.Nets, func(i, j int) bool {
		return g.Nets[i].Nols < g.Nets[j].Nols
	})
	return g
}

func (g *Genetic) showList(max ...int) {
	for i, el := range g.Nets {
		if len(max) > 0 && max[0] >= i {
			log.Println(i, el.Result)
		}
	}
}

func (r *ResOrder) CalcSum(budget float64, inpData []DataTeach) {
	r.Score = 0
	r.Sum = budget
	for _, t := range r.Trades {
		if !r.Type {
			r.Sum -= inpData[t].Price
			r.Type = true
		} else {
			r.Sum += inpData[t].Price
			r.Type = false
		}
	}
	r.Score = r.Sum
}

func (g *Genetic) TrainItemOrders() {
	var wg sync.WaitGroup
	for _, res := range g.ResOrders {
		wg.Add(1)
		go func(r *ResOrder) {
			defer wg.Done()
			r.Type = false
			r.Diff = 0
			r.Score = 0
			r.Sum = g.Config.Budget
			for _, t := range r.Trades {
				if !r.Type {
					r.Sum -= g.Config.DataList[t].Price
					r.Type = true
				} else {
					r.Sum += g.Config.DataList[t].Price
					r.Type = false
				}
			}
		}(res)
	}
	wg.Wait()
}

func (g *Genetic) TrainItem(ret func(n *NetPerc)) {
	var wg sync.WaitGroup
	for i, _ := range g.Nets {
		wg.Add(1)
		go func(ind int) {
			defer wg.Done()
			g.Nets[ind].Trades = 0
			g.Nets[ind].Nols = 0
			g.Nets[ind].Score = 0
			g.Nets[ind].Scores = nil
			g.Nets[ind].Utilses = nil
			g.Nets[ind].Budget = g.Config.Budget
			g.Nets[ind].LastPrice = 0
			g.Nets[ind].DiffPerce = 0
			g.Nets[ind].StatusBSell = false
			ret(g.Nets[ind])
		}(i)
	}
	wg.Wait()
}

func (g *Genetic) Train(last bool) {
	var wg sync.WaitGroup
	wg.Add(len(g.Nets))
	for i, _ := range g.Nets {
		go func(ind int) {
			defer wg.Done()
			g.Nets[ind].TrainIter()
		}(i)
	}
	wg.Wait()

	g.sortBest()
	g.sliceBest()

	if !last {
		g.mutate()
	}

	g.Score = g.GetBest().Score
}

func (g *Genetic) CheckScore() bool {
	return g.Score >= g.Config.BestResult
}

func (g *Genetic) LogScore(i int) {
	if g.Iters%i == 0 {
		log.Println(
			g.Iters, " - iter; ",
			"MAX:", toFixed(g.Nets[0].Budget, 3),
			"MIN:", toFixed(g.Nets[len(g.Nets)-1].Budget, 3), "; ",
			"len:", len(g.Nets), "; ",
			"score:", fmt.Sprintf("%.3f", g.GetBest().Score), "; ",
			"trades:", g.GetBest().Trades, "; ",
			"diff:", toFixed(g.GetBest().DiffPerce, 3),
		)
	}
}

func (g *Genetic) LogScoreOrders(i int, str ...string) {
	if g.Iters%i == 0 {
		var stItem string
		if len(str) > 0 {
			stItem = str[0]
		}
		log.Println(
			g.Iters, " - iter; ",
			stItem+"; ",
			"SCORE:", g.GetBestOrders().Score, "; ",
			"COUNT:", g.GetBestOrders().Count, "; ",
			"SUM:", g.GetBestOrders().Sum, "; ",
			"DIFF:", fmt.Sprintf("%.2f", g.GetBestOrders().Diff), "; ",
			"LENGTH:", len(g.ResOrders), "; ",
		)
	}
}

func (g *Genetic) LogScoreNols(i int) {
	if g.Iters%i == 0 {
		log.Println(g.Iters, " - iter; ", g.Nets[0].Nols, " - nols")
	}
}

func (g *Genetic) sortBestNolsIters() {

}

func (g *Genetic) Iterate() bool {
	g.sortBest()
	g.sliceBest()
	g.Score = g.GetBest().Budget
	//g.mutate()
	g.mutateV2()
	g.Iters += 1
	return false
}

func (g *Genetic) IterateOrders() {
	g.sortBestOrders()
	g.sliceBestOrders()
	g.Score = g.GetBestOrders().Sum
}

func (g *Genetic) IterateNols(lastNols int) bool {
	g.sortBestNols()
	g.sliceBest()
	var stats int
	for i := 0; i < lastNols; i++ {
		if g.Nets[i].Nols == 0 {
			stats += 1
		}
	}
	if stats == lastNols {
		return true
	}
	g.Iters += 1
	g.mutate()
	return false
}

func (g *Genetic) sliceBest() {
	if len(g.Nets) > g.Config.LastBest {
		g.Nets = g.Nets[:g.Config.LastBest]
	}
}

func (g *Genetic) sliceBestOrders() {
	if len(g.ResOrders) > g.Config.LastBest {
		g.ResOrders = g.ResOrders[:g.Config.LastBest]
	}
}

func (g *Genetic) mutate() {
	var (
		wg          sync.WaitGroup
		mtWait      sync.Mutex
		listNetsAdd []*NetPerc
	)
	ind := len(g.Nets) - 1
	if ind == 0 {
		for s := ind; s < g.Config.Population; s++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				n := g.Nets[0].Copy()
				var wgw sync.WaitGroup
				for i := 0; i < g.Config.LimitMutateSub; i++ {
					wgw.Add(1)
					go func(nn *NetPerc) {
						defer wgw.Done()
						nn.mutateWeight(g.Config.MinRandWeight, g.Config.MaxRandWeight)
					}(n)
				}
				wgw.Wait()
				mtWait.Lock()
				listNetsAdd = append(listNetsAdd, n)
				mtWait.Unlock()

			}()
		}
	} else {
		for s := ind; s < g.Config.Population; s++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r := randIntMin(1, ind)
				n := g.Nets[r].Copy()
				var wgw sync.WaitGroup
				for i := 0; i < g.Config.LimitMutateSub; i++ {
					wgw.Add(1)
					go func(nn *NetPerc) {
						defer wgw.Done()
						nn.mutateWeight(g.Config.MinRandWeight, g.Config.MaxRandWeight)
					}(n)
				}
				wgw.Wait()
				mtWait.Lock()
				listNetsAdd = append(listNetsAdd, n)
				mtWait.Unlock()

			}()
		}
	}
	wg.Wait()
	g.Nets = append(g.Nets, listNetsAdd...)
}

func (g *Genetic) Save(fileName string) error {
	if fileName == "" {
		return errors.New("empty filename")
	}
	if bts, err := json.Marshal(g); err != nil {
		return err
	} else {
		if err := ioutil.WriteFile(fileName, bts, 0644); err != nil {
			return err
		}
	}
	return nil
}

func (g *Genetic) Load(fileName string) bool {
	if fileName == "" {
		log.Println("empty filename")
		return false
	}
	bts, err := ioutil.ReadFile(fileName)
	if err != nil {
		log.Println(err)
		return false
	}
	if err := json.Unmarshal(bts, &g); err != nil {
		log.Println(err)
		return false
	}
	return true
}

func LoadGen(fileName string) (*Genetic, bool) {
	var gen Genetic
	if fileName == "" {
		log.Println("empty filename")
		return &gen, false
	}
	bts, err := ioutil.ReadFile(fileName)
	if err != nil {
		log.Println(err)
		return &gen, false
	}
	if err := json.Unmarshal(bts, &gen); err != nil {
		log.Println(err)
		return &gen, false
	}
	return &gen, true
}

func (g *Genetic) mutateV2() {
	var (
		wg          sync.WaitGroup
		mtWait      sync.Mutex
		listNetsAdd []*NetPerc
	)
	ind := len(g.Nets) - 1
	for s := ind; s < g.Config.Population; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			n := g.Nets[0].Copy()
			n.mutateWeight(g.Config.MinRandWeight, g.Config.MaxRandWeight)
			mtWait.Lock()
			listNetsAdd = append(listNetsAdd, n)
			mtWait.Unlock()
		}()
	}
	wg.Wait()
	g.Nets = append(g.Nets, listNetsAdd...)
}

func (g *Genetic) MutateOrdersV3() {
	for i := 0; len(g.ResOrders) < g.Config.Population; i++ {
		next := g.ResOrders[i].CopyOrig()
		for s := 0; s < g.Config.MaxMutateIter; s++ {
			next.mutateV3(g.Config.Inps)
		}
		g.ResOrders = append(g.ResOrders, next)
		if i >= g.Config.LastBest {
			i = 0
		}
	}
}

func (g *Genetic) mutateOrders() {
	var (
		wg         sync.WaitGroup
		mtWait     sync.Mutex
		listOrders []*ResOrder
	)
	ind := len(g.ResOrders) - 1
	for s := ind; s < g.Config.Population; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rnd := randIntMin(0, ind)
			r := g.ResOrders[rnd].Copy()
			r.mutate()
			mtWait.Lock()
			listOrders = append(listOrders, r)
			mtWait.Unlock()
		}()
	}
	wg.Wait()
	for i := 0; i < g.Config.NewItems; i++ {
		g.ResOrders = append(g.ResOrders, g.AddOrder())
	}
	g.ResOrders = append(g.ResOrders, listOrders...)
}

func (r *ResOrder) Copy() *ResOrder {
	rr := ResOrder{
		Count: r.Count,
		By:    r.By,
		Type:  false,
		Sum:   r.Sum,
		Score: r.Score,
	}
	rr.Trades = make([]int, len(r.Trades))
	copy(rr.Trades, r.Trades)
	return &rr
}

func (r *ResOrder) CopyJson() *ResOrder {
	var rr ResOrder
	bts, _ := json.Marshal(r)
	json.Unmarshal(bts, &rr)
	rr.Type = false
	return &rr
}

func (r *ResOrder) GetBytesJson() []byte {
	bts, err := json.Marshal(r)
	if err != nil {
		log.Println("error marshal ResOrder: ", err)
	}
	return bts
}

func (r *ResOrder) CopyOrig() *ResOrder {
	rr := ResOrder{Count: r.Count}
	rr.Trades = make([]int, len(r.Trades))
	copy(rr.Trades, r.Trades)
	return &rr
}

func remove(s []int, i int) []int {
	s[i] = s[len(s)-1]
	return s[:len(s)-1]
}

func (r *ResOrder) mutateAll(max int) {
	mp := make(map[int]struct{})
	trds := []int{}
	for {
		in := funk.RandomInt(0, max)
		if _, ok := mp[in]; ok {
			continue
		}
		mp[in] = struct{}{}
		trds = append(trds, in)
		if len(trds) >= r.Count {
			break
		}
	}
	sort.Slice(trds, func(i, j int) bool {
		return trds[i] < trds[j]
	})
	r.Trades = trds
}

func (r *ResOrder) copyAndMutate(maxVal int) *ResOrder {
	var res *ResOrder
	bts, err := json.Marshal(r)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(bts, &res); err != nil {
		log.Fatal(err)
	}
	return res.mutateV3(maxVal)
}

func (g *Genetic) FirstMutateOrders(iters int) {
	g.LBOitem = &LBO{Score: -1, Count: 0, Trades: []int{}}
	for i := 0; i < iters; i++ {
		var wg sync.WaitGroup
		wg.Add(len(g.ResOrders))
		for _, ord := range g.ResOrders {
			func(res *ResOrder) {
				defer wg.Done()
				res.mutateV3(g.Config.Inps)
				res.CalcSum(g.Config.Budget, g.Config.DataList)
			}(ord)
		}
		wg.Wait()
		g.SortResOrdersV2()
		if g.LBOitem.Score < g.GetBestOrders().Score {
			lbo := LBO{
				Score:  g.GetBestOrders().Score,
				Count:  g.GetBestOrders().Count,
				Trades: g.GetBestOrders().Trades,
			}
			g.LBOitem = &lbo
			fmt.Printf("iteration: %.f; best result: %.3f; count - %.f; sum: %.3f\n", float64(i), g.GetBestOrders().Score, float64(g.GetBestOrders().Count), g.GetBestOrders().Sum)
		}
		//fmt.Printf("best result: %.3f; count - %.f; sum: %.3f\n", g.GetBestOrders().Score, float64(g.GetBestOrders().Count), g.GetBestOrders().Sum)
	}
}

func (r *ResOrder) mutateV3(maxVal int) *ResOrder {
	rnd := funk.RandomInt(0, len(r.Trades))
	var nVal int
	switch rnd {
	case 0:
		nVal = funk.RandomInt(0, r.Trades[1])
	case len(r.Trades) - 1:
		nVal = funk.RandomInt(r.Trades[len(r.Trades)-2], maxVal)
	default:
		if r.Trades[rnd-1]+1 < r.Trades[rnd+1] {
			nVal = funk.RandomInt(r.Trades[rnd-1]+1, r.Trades[rnd+1])
		} else {
			nVal = r.Trades[rnd-1] + 1
		}
	}
	r.Trades[rnd] = nVal
	return r
}

func (r *ResOrder) mutate() {
	var nVal int
	rand := randIntMin(0, len(r.Trades)-1)
	if rand == 0 {
		if len(r.Trades) != 1 {
			nVal = randIntMin(0, r.Trades[rand+1]-1)
		}
	}
	if rand == len(r.Trades)-1 {
		if len(r.Trades) != 1 {
			nVal = randIntMin(r.Trades[rand-1], r.Trades[rand])
		}
	}
	if rand != 0 && rand != len(r.Trades)-1 {
		if len(r.Trades) != 1 {
			nVal = randIntMin(r.Trades[rand-1], r.Trades[rand+1])
		}
	}
	r.Trades[rand] = nVal
}

func (g *Genetic) mutateOrdersFirst() {
	var wg sync.WaitGroup
	wg.Add(len(g.ResOrders[1:]))
	for _, ord := range g.ResOrders[1:] {
		go func(ordItem *ResOrder) {
			defer wg.Done()
			ordItem.mutateAll(g.Config.Inps)
			/*for i := 0; i < ordItem.Count; i++ {
				ordItem.mutateV3(g.Config.Inps)
			}*/
		}(ord)
	}
	wg.Wait()
}

func (g *Genetic) GenerateOrdersV6(maxIter int) *Genetic {
	minIter := g.Config.Inps / 100 * 10
	for d := minIter; d < minIter+maxIter; d++ {
		order := ResOrder{Count: d}
		order.GenerateTradesV2(g.Config.DataList)
		g.ResOrders = append(g.ResOrders, &order)
	}
	for _, o := range g.ResOrders {
		if o.Count%2 != 0 {
			o.Trades = o.Trades[1:]
			o.Count = len(o.Trades)
		}
	}
	return g
}

func (g *Genetic) CalcSumOrder(ord *ResOrder) {
	ord.OldSum = ord.Sum
	ord.Sum = 0
	ord.Type = false
	for _, i := range ord.Trades {
		if !ord.Type {
			ord.Sum -= g.Config.DataList[i].Price
			ord.Type = true
		} else {
			ord.Sum += g.Config.DataList[i].Price
			ord.Type = false
		}
	}
}

func (ord *ResOrder) SumCheck(list []DataTeach) {
	ord.OldSum = ord.Sum
	ord.Sum = 0
	ord.Type = false
	for _, i := range ord.Trades {
		if !ord.Type {
			ord.Sum -= list[i].Price
			ord.Type = true
		} else {
			ord.Sum += list[i].Price
			ord.Type = false
		}
	}
}

func (g *Genetic) TrainOrders(maxAgain int) {
	mpData := map[float64]int{}
	for m := 0; true; m++ {
		// ==============================
		var wg sync.WaitGroup
		wg.Add(len(g.ResOrders))
		for _, ord := range g.ResOrders {
			go func(o *ResOrder) {
				defer wg.Done()
				g.CalcSumOrder(o)
			}(ord)
		}
		wg.Wait()
		// ==============================
		sort.Slice(g.ResOrders, func(i, j int) bool {
			return g.ResOrders[i].Sum > g.ResOrders[j].Sum
		})
		// ==============================
		g.ResOrders = g.ResOrders[:g.Config.LastBest]
		// ==============================
		if _, ok := mpData[g.GetBestOrders().Sum]; !ok {
			mpData[g.GetBestOrders().Sum] = 0
		}
		mpData[g.GetBestOrders().Sum] += 1
		// ==============================
		if m%100 == 0 {
			fmt.Printf("%.3f\n", g.GetBestOrders().Sum)
		}
		// ==============================
		if mpData[g.GetBestOrders().Sum] > maxAgain {
			break
		}
		//tm1 := time.Now()
		// ==============================
		g.MutateOrdersV5()
		// ==============================
		//println(time.Now().Sub(tm1).String())
	}
}

func (g *Genetic) MutateOrdersV5() {
	for {
		var wg sync.WaitGroup
		var mt sync.Mutex
		var list []*ResOrder
		for i := 0; i < len(g.ResOrders); i++ {
			wg.Add(1)
			go func(ord *ResOrder) {
				defer wg.Done()
				oCopy := ord.CopyOrig()
				oCopy.MutaterOrderV2(g.Config.DataList)
				mt.Lock()
				list = append(list, oCopy)
				mt.Unlock()
			}(g.ResOrders[i])
		}
		wg.Wait()
		if len(list) > 0 {
			g.ResOrders = append(g.ResOrders, list...)
		}
		if len(g.ResOrders) > g.Config.Population {
			g.ResOrders = g.ResOrders[:g.Config.Population]
			break
		}
	}
}

func removeSl(slice []int, s int) []int {
	return append(slice[:s], slice[s+1:]...)
}

func (ord *ResOrder) MutaterOrderV2(list []DataTeach) {
	cnt := randInt(ord.Count)
	mpData := map[int]bool{}
	for _, tr := range ord.Trades {
		mpData[tr] = true
	}
	for {
		rNew := randIntMin(0, len(list)-1)
		if _, ok := mpData[rNew]; !ok {
			ord.Trades[cnt] = rNew
			break
		}
	}
	sort.Ints(ord.Trades)
}

func (g *Genetic) MutateOrdersV4() {
	var wg sync.WaitGroup
	wg.Add(len(g.ResOrders))
	for _, ord := range g.ResOrders {
		go func(o *ResOrder) {
			defer wg.Done()
			o.MutaterOrderV1(g.Config.DataList)
		}(ord)
	}
	wg.Wait()
}

func (o *ResOrder) MutaterOrderV1(list []DataTeach) {
	oCopy := o.CopyOrig()
	cnt := randInt(oCopy.Count)
	min, max := func() (int, int) {
		if cnt == 0 {
			return 0, oCopy.Trades[1] - 1
		}
		if cnt >= oCopy.Count-1 {
			return oCopy.Trades[cnt-1] + 1, oCopy.Trades[oCopy.Count-1]
		}
		return oCopy.Trades[cnt-1] + 1, oCopy.Trades[cnt+1] - 1
	}()
	oCopy.Trades[cnt] = randIntMin(min, max)
	oCopy.SumCheck(list)
	if o.Sum < oCopy.Sum {
		copy(o.Trades, oCopy.Trades)
	}
}

func (g *Genetic) GenerateOrdersDownV7(maxIter int) *Genetic {
	minIter := g.Config.Inps / 100 * 2
	for d := minIter; d < minIter+maxIter; d++ {
		order := ResOrder{Count: d}
		order.GenerateTradesV2(g.Config.DataList)
		g.ResOrders = append(g.ResOrders, &order)
	}
	return g
}

func (g *Genetic) TrainOrdersDown(iters int) {

	for i := 0; i < iters; i++ {
		// ==============================
		var wg sync.WaitGroup
		wg.Add(len(g.ResOrders))
		for _, ord := range g.ResOrders {
			go func(o *ResOrder) {
				defer wg.Done()
				g.CalcDownOrder(o)
			}(ord)
		}
		wg.Wait()
		// ==============================
		sort.Slice(g.ResOrders, func(i, j int) bool {
			return g.ResOrders[i].Sum < g.ResOrders[j].Sum
		})
		// ==============================
		if i%10000 == 0 {
			fmt.Printf("SUM: %.3f\n", g.GetBestOrders().Sum)
		}
		// ==============================
		g.MutateOrdersDownV5()
		// ==============================

	}

}

func (g *Genetic) CalcDownOrder(ord *ResOrder) {
	ord.OldSum = ord.Sum
	ord.Sum = 0
	for _, i := range ord.Trades {
		ord.Sum -= g.Config.DataList[i].Price
	}
}

func (ord *ResOrder) CalcDownOrder(list []DataTeach) {
	ord.OldSum = ord.Sum
	ord.Sum = 0
	for _, i := range ord.Trades {
		if !ord.Type {
			ord.Sum -= list[i].Price
			ord.Type = true
		} else {
			ord.Sum += list[i].Price
			ord.Type = false
		}
	}
}

func (g *Genetic) MutateOrdersDownV5() {
	var wg sync.WaitGroup
	wg.Add(len(g.ResOrders))
	for _, ord := range g.ResOrders {
		go func(o *ResOrder) {
			defer wg.Done()
			o.MutaterOrderDownV2(g.Config.DataList)
		}(ord)
	}
	wg.Wait()
}

func (o *ResOrder) MutaterOrderDownV2(list []DataTeach) {
	oCopy := o.CopyOrig()
	cnt := randInt(oCopy.Count)
	min, max := func() (int, int) {
		if cnt == 0 {
			return 0, oCopy.Trades[1] - 1
		}
		if cnt >= oCopy.Count-1 {
			return oCopy.Trades[cnt-1] + 1, oCopy.Trades[oCopy.Count-1]
		}
		return oCopy.Trades[cnt-1] + 1, oCopy.Trades[cnt+1] - 1
	}()
	oCopy.Trades[cnt] = randIntMin(min, max)
	oCopy.SumCheck(list)
	if o.Sum < oCopy.Sum {
		copy(o.Trades, oCopy.Trades)
	}
}

func (g *Genetic) GenerateOrdersV7() *Genetic {
	minIter := g.Config.Inps / 50
	maxIter := g.Config.Inps / 10

	log.Println("MIN:", minIter, "; MAX:", maxIter)

	for d := minIter; d < maxIter; d++ {
		order := ResOrder{Count: d}
		order.GenerateTradesV2(g.Config.DataList)
		g.ResOrders = append(g.ResOrders, &order)
		if d+1 >= maxIter {
			d = 0
		}
		if len(g.ResOrders) >= g.Config.Population {
			break
		}
	}
	return g
}

func (ord *ResOrder) GenerateTradesV2(list []DataTeach) {
	mpData := map[int]bool{}
	for {
		cnt := randIntMin(0, len(list))
		if _, ok := mpData[cnt]; !ok {
			mpData[cnt] = true
			ord.Trades = append(ord.Trades, cnt)
			if len(ord.Trades) >= ord.Count {
				break
			}
		}
	}
	sort.Ints(ord.Trades)
}

func (g *Genetic) Tuning(max int) {
	for i := 0; i < max; i++ {
		mpData := map[int]bool{}
		ord := g.GetBestOrders().CopyOrig()
		ord.SumCheck(g.Config.DataList)
		for _, tr := range ord.Trades {
			mpData[tr] = true
		}
		rndKey := randInt(ord.Count)
		var minSum, maxSum float64
		minVl, maxVl := ord.Trades[rndKey]-1, ord.Trades[rndKey]+1
		if minVl < 0 {
			minVl = 0
		}
		if maxVl > g.Config.Inps-1 {
			maxVl = g.Config.Inps - 1
		}
		_, okMin := mpData[minVl]
		_, okMax := mpData[maxVl]
		if !okMin {
			minOrd := ord.CopyOrig()
			minOrd.Trades[rndKey] = minVl
			minOrd.SumCheck(g.Config.DataList)
			minSum = minOrd.Sum
		}
		if !okMax {
			maxOrd := ord.CopyOrig()
			maxOrd.Trades[rndKey] = maxVl
			maxOrd.SumCheck(g.Config.DataList)
			maxSum = maxOrd.Sum
		}
		if ord.Sum > minSum && ord.Sum < maxSum {
			continue
		}
		if ord.Sum < minSum {
			g.GetBestOrders().Trades[rndKey] = minVl
			g.GetBestOrders().SumCheck(g.Config.DataList)
			continue
		}
		if ord.Sum < maxSum {
			g.GetBestOrders().Trades[rndKey] = maxVl
			g.GetBestOrders().SumCheck(g.Config.DataList)
			continue
		}
		if i%10000 == 0 {
			fmt.Printf("SUM: %.3f \n", ord.Sum)
		}
	}
}

func GeneticFormed(list []DataTeach) *Genetic {
	// ====================
	gen := InitGenetic(GeneticConf{
		Population: 10000,
		LastBest:   100,
		DataList:   list,
		Inps:       len(list),
	})
	// ====================
	gen.GenerateOrdersV7()
	gen.TrainOrders(1000)
	// ====================
	return gen
}

const fileKlinesAll = "klinesAll.dtx"

func KLineListSave(list interface{}) {
	bts, err := json.Marshal(list)
	if err != nil {
		log.Println("error all klines save:", err)
		return
	}
	if err := ioutil.WriteFile(fileKlinesAll, bts, 0644); err != nil {
		log.Println("error all klines file save:", err)
		return
	}
}

func KLineListLoad(in interface{}) bool {
	bts, err := ioutil.ReadFile(fileKlinesAll)
	if err != nil {
		log.Println("error all klines read file:", err)
		return false
	}
	if err := json.Unmarshal(bts, in); err != nil {
		log.Println("error all klines unmarshal:", err)
		return false
	}
	return true
}
