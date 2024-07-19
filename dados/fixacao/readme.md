`batteryRUL.csv`: **dicionário de dados**

| Feature                 | Description                |
|-------------------------|----------------------------|
| cycle_index             | number of cycle            |
| discharge_time_s        | Discharge Time (s)         |
| decrement               | Decrement 3.6-3.4V (s)     |
| max_voltage_discharge   | Max. Voltage Discharge (V) |
| min_voltage_charge      | Min. Voltage Discharge (V) |
| time_4v                 | Time at 4.15V (s)          |
| time_constant_current_s | Time Constant Current (s)  |
| charging_time_s         | Charging Time (s)          |
| remaining_useful_life   | target (RUL)               |

<br>

`credit_risk.csv`: **dicionário de dados**

|          Feature |                    Description |
|-----------------:|-------------------------------:|
|      customer_id | customer id                    |
|      overdue_sum | total overdue days             |
|       pay_normal | number of times normal payment |
|     credit_limit | credit   limit                 |
|      new_balance | current balance                |
|  highest_balance | highest balance in history     |
| high_credit_risk | 1 if high credit risk, else 0  |

<br>

`mall_customers.csv`: **dicionário de dados**

|          Feature |                    Description |
|-----------------:|-------------------------------:|
|      id          | customer id                    |
|      age         | customer age                   |
|      income      | customer's annual income       |
|     score        | buying potential               |


EXTRA: [Compilado de Pandas para manipulação de dados](https://oviedovr.github.io/compilado-pandas/)
