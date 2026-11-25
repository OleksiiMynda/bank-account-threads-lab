"""
Моделювання банківського рахунку з одночасним доступом декількох клієнтів.

Структура відповідає завданню:
  1. BankAccount          - початковий баланс, поповнення/зняття, перевірка
                            достатності коштів, журнал транзакцій.
  2. client_worker        - потік-клієнт, що генерує випадкові операції;
                            усі потоки стартують одночасно (Barrier).
  3. UnsafeBankAccount    - обробка БЕЗ синхронізації (гонка даних).
  4. SafeBankAccount      - захист через threading.Lock (атомарні
                            "перевірка + зміна + запис у журнал").
  5. experiment_*         - порівняння версій, залежність від кількості
                            потоків/транзакцій, вплив блокувань на час.

Запуск:
    python bank_simulation.py            # повний набір експериментів
    python bank_simulation.py --quick    # скорочений варіант
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEPOSIT = "DEPOSIT"
WITHDRAW = "WITHDRAW"


@dataclass
class Transaction:
    seq: int            # порядковий номер у журналі
    client: int         # номер клієнта (потоку)
    kind: str           # DEPOSIT / WITHDRAW
    amount: int
    ok: bool            # чи виконана операція (для зняття - чи вистачило коштів)
    balance_after: int  # баланс, який записала операція


class BankAccount:
    """Базовий рахунок. Підкласи визначають, як саме захищено баланс.

    latency=True імітує "час обробки" запиту (звернення до БД, мережі) між
    читанням балансу та записом нового значення: потік віддає процесор
    (time.sleep(0)). Саме в цьому вікні і виникає гонка даних. Без нього, через
    GIL, у CPython гонка трапляється дуже рідко, хоча помилка в коді та сама.
    """

    def __init__(self, initial_balance: int, latency: bool = True):
        self._balance = initial_balance
        self.initial_balance = initial_balance
        self.latency = latency
        self.log: list[Transaction] = []

    @property
    def balance(self) -> int:
        return self._balance

    def deposit(self, client: int, amount: int) -> bool:
        raise NotImplementedError

    def withdraw(self, client: int, amount: int) -> bool:
        raise NotImplementedError

    def _processing_delay(self) -> None:
        if self.latency:
            time.sleep(0)  # віддати квант часу іншому потоку

    def _record(self, client: int, kind: str, amount: int, ok: bool, balance_after: int) -> None:
        self.log.append(Transaction(len(self.log), client, kind, amount, ok, balance_after))


class UnsafeBankAccount(BankAccount):
    """Два класичні дефекти без захисту:
      * поповнення: read -> (затримка) -> write  => ВТРАЧЕНІ ОНОВЛЕННЯ
        (запис затирає зміни, зроблені іншими потоками за цей час);
      * зняття: check -> (затримка) -> змінa     => TOCTOU: коли баланс
        списується, перевірка вже застаріла, тому рахунок може піти в мінус."""

    def deposit(self, client: int, amount: int) -> bool:
        current = self._balance                 # читання
        self._processing_delay()
        self._balance = current + amount        # запис (міг "затерти" чужі зміни)
        self._record(client, DEPOSIT, amount, True, self._balance)
        return True

    def withdraw(self, client: int, amount: int) -> bool:
        if self._balance < amount:              # перевірка достатності коштів
            self._record(client, WITHDRAW, amount, False, self._balance)
            return False
        self._processing_delay()                # ...інші потоки встигають зняти гроші
        self._balance -= amount                 # списання за вже застарілою перевіркою
        self._record(client, WITHDRAW, amount, True, self._balance)
        return True


class SafeBankAccount(BankAccount):
    """Уся критична секція (читання, перевірка, зміна, запис у журнал) виконується
    під одним замком, тому перевірка достатності коштів і списання атомарні."""

    def __init__(self, initial_balance: int, latency: bool = True):
        super().__init__(initial_balance, latency)
        self._lock = threading.Lock()

    def deposit(self, client: int, amount: int) -> bool:
        with self._lock:
            current = self._balance
            self._processing_delay()
            self._balance = current + amount
            self._record(client, DEPOSIT, amount, True, self._balance)
            return True

    def withdraw(self, client: int, amount: int) -> bool:
        with self._lock:
            current = self._balance
            if current < amount:
                self._record(client, WITHDRAW, amount, False, current)
                return False
            self._processing_delay()
            self._balance = current - amount
            self._record(client, WITHDRAW, amount, True, self._balance)
            return True


@dataclass
class ClientResult:
    deposited: int = 0       # сума успішних поповнень (з точки зору клієнта)
    withdrawn: int = 0       # сума зняттів, які клієнт отримав як "успішні"
    rejected: int = 0        # кількість відмов через брак коштів


def generate_operations(seed: int, count: int, max_amount: int, p_deposit: float):
    """Детермінований набір операцій клієнта: однаковий для обох версій,
    тому порівняння коректне."""
    rng = random.Random(seed)
    return [
        (DEPOSIT if rng.random() < p_deposit else WITHDRAW, rng.randint(1, max_amount))
        for _ in range(count)
    ]


def client_worker(account: BankAccount, client_id: int, ops, barrier: threading.Barrier,
                  result: ClientResult) -> None:
    barrier.wait()  # усі клієнти починають одночасно
    for kind, amount in ops:
        if kind == DEPOSIT:
            account.deposit(client_id, amount)
            result.deposited += amount
        else:
            if account.withdraw(client_id, amount):
                result.withdrawn += amount
            else:
                result.rejected += 1


@dataclass
class RunStats:
    version: str
    threads: int
    ops_per_thread: int
    latency: bool
    initial: int
    expected: int
    actual: int
    elapsed: float
    rejected: int
    negative_events: int      # записи журналу з від'ємним балансом
    log_inconsistencies: int  # розриви ланцюжка "баланс до -> баланс після"
    min_balance: int = 0

    @property
    def error(self) -> int:
        return self.actual - self.expected

    @property
    def incorrect(self) -> bool:
        return self.error != 0


def analyse_log(account: BankAccount):
    """Перевіряє журнал: чи кожна успішна операція змінила попередній баланс
    рівно на свою суму. Для коректної версії порушень немає."""
    prev = account.initial_balance
    broken = negative = 0
    min_bal = prev
    for t in account.log:
        if t.ok:
            delta = t.amount if t.kind == DEPOSIT else -t.amount
            if t.balance_after != prev + delta:
                broken += 1
            prev = t.balance_after
        if t.balance_after < 0:
            negative += 1
        min_bal = min(min_bal, t.balance_after)
    return broken, negative, min_bal


def run_simulation(safe: bool, threads: int, ops_per_thread: int, *, initial: int = 1000,
                   max_amount: int = 100, p_deposit: float = 0.5, latency: bool = True,
                   seed: int = 0) -> RunStats:
    cls = SafeBankAccount if safe else UnsafeBankAccount
    account = cls(initial, latency)
    results = [ClientResult() for _ in range(threads)]
    barrier = threading.Barrier(threads + 1)
    workload = [generate_operations(seed * 10_000 + i, ops_per_thread, max_amount, p_deposit)
                for i in range(threads)]
    pool = [threading.Thread(target=client_worker, args=(account, i, workload[i], barrier, results[i]))
            for i in range(threads)]
    for t in pool:
        t.start()
    barrier.wait()
    start = time.perf_counter()
    for t in pool:
        t.join()
    elapsed = time.perf_counter() - start

    expected = initial + sum(r.deposited for r in results) - sum(r.withdrawn for r in results)
    broken, negative, min_bal = analyse_log(account)
    return RunStats("з Lock" if safe else "без синхр.", threads, ops_per_thread, latency, initial,
                    expected, account.balance, elapsed, sum(r.rejected for r in results),
                    negative, broken, min_bal)


def average(runs: list[RunStats]) -> dict:
    return {
        "error_mean": statistics.mean(abs(r.error) for r in runs),
        "wrong_runs": sum(r.incorrect for r in runs),
        "negative": statistics.mean(r.negative_events for r in runs),
        "broken": statistics.mean(r.log_inconsistencies for r in runs),
        "time": statistics.mean(r.elapsed for r in runs),
    }


def table(headers, rows) -> str:
    cols = list(zip(headers, *rows))
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:>{w}}}" for w in widths) + " |"
    out = [line, fmt.format(*headers), line]
    out += [fmt.format(*map(str, r)) for r in rows]
    out.append(line)
    return "\n".join(out)


def title(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def experiment_basic(threads: int, ops: int) -> None:
    title(f"Пп. 3-5. Одиночний запуск обох версій: {threads} потоків x {ops} операцій, "
          f"початковий баланс 1000")
    rows = []
    for safe in (False, True):
        s = run_simulation(safe, threads, ops, seed=1)
        rows.append([s.version, s.initial, s.expected, s.actual, f"{s.error:+d}",
                     s.negative_events, s.log_inconsistencies, s.rejected, f"{s.elapsed:.3f}"])
    print(table(["Версія", "Початк.", "Очікуваний", "Фактичний", "Похибка",
                 "Від'ємні", "Розриви журн.", "Відмов", "Час, с"], rows))
    print("Очікуваний = початковий + сума поповнень - сума зняттів, які клієнти "
          "отримали як успішні.\n'Від'ємні' - записи журналу з балансом < 0 "
          "(зняття, що пройшло перевірку на застарілому значенні).")


def experiment_gil_note(threads: int, ops: int, repeats: int) -> None:
    title("Довідково. Вплив 'вікна гонки' (затримки між read і write) на помилки без синхронізації")
    rows = []
    for latency in (False, True):
        runs = [run_simulation(False, threads, ops, latency=latency, seed=s) for s in range(repeats)]
        a = average(runs)
        rows.append(["так" if latency else "ні", f"{a['wrong_runs']}/{repeats}",
                     f"{a['error_mean']:.0f}", f"{a['negative']:.1f}", f"{a['time']:.3f}"])
    print(table(["Затримка між read/write", "Запусків з помилкою", "Сер. модуль похибки",
                 "Сер. від'ємних", "Сер. час, с"], rows))
    print("Через GIL інтерпретатор перемикає потоки рідко (раз на ~5 мс), тому без затримки "
          "гонка може не проявитись узагалі,\nале помилка в алгоритмі лишається: у реальній "
          "системі між читанням і записом є ввід/вивід, мережа, БД.")


def experiment_scaling(vary: str, values, fixed: int, repeats: int, rows_csv: list) -> None:
    label = "кількості ПОТОКІВ (операцій на потік = %d)" % fixed if vary == "threads" \
        else "кількості ОПЕРАЦІЙ на потік (потоків = %d)" % fixed
    title(f"П. 5. Залежність від {label}; усереднення по {repeats} запусках")
    rows = []
    for v in values:
        th, ops = (v, fixed) if vary == "threads" else (fixed, v)
        un = average([run_simulation(False, th, ops, seed=s) for s in range(repeats)])
        sf = average([run_simulation(True, th, ops, seed=s) for s in range(repeats)])
        rows.append([v, f"{un['wrong_runs']}/{repeats}", f"{un['error_mean']:.0f}",
                     f"{un['negative']:.1f}", f"{un['broken']:.0f}",
                     f"{sf['wrong_runs']}/{repeats}", f"{sf['error_mean']:.0f}",
                     f"{sf['negative']:.1f}"])
        rows_csv.append([vary, v, un["wrong_runs"], un["error_mean"], un["negative"], un["broken"],
                         sf["wrong_runs"], sf["error_mean"], sf["negative"]])
    hdr = ["Потоків" if vary == "threads" else "Операцій/потік",
           "БЕЗ: помилк.запусків", "БЕЗ: сер.мод.похибки", "БЕЗ: від'ємних", "БЕЗ: розриви",
           "Lock: помилк.запусків", "Lock: сер.мод.похибки", "Lock: від'ємних"]
    print(table(hdr, rows))


def experiment_timing(thread_counts, ops: int, repeats: int, rows_csv: list) -> None:
    title(f"П. 5. Вплив блокувань на час виконання ({ops} операцій на потік, "
          f"{repeats} повторів)")
    rows = []
    for latency in (True, False):
        for th in thread_counts:
            un = average([run_simulation(False, th, ops, latency=latency, seed=s) for s in range(repeats)])["time"]
            sf = average([run_simulation(True, th, ops, latency=latency, seed=s) for s in range(repeats)])["time"]
            total = th * ops
            rows.append(["так" if latency else "ні", th, total, f"{un:.3f}", f"{sf:.3f}",
                         f"{sf / un:.2f}x", f"{total / un:,.0f}", f"{total / sf:,.0f}"])
            rows_csv.append([latency, th, total, un, sf, sf / un])
    print(table(["Затримка", "Потоків", "Всього оп.", "БЕЗ, с", "Lock, с", "Lock/БЕЗ",
                 "БЕЗ, оп/с", "Lock, оп/с"], rows))
    print("Lock/БЕЗ > 1 - вартість коректності. У CPython (GIL) потоки й так не "
          "виконують Python-код паралельно,\nтому основна ціна замка - накладні "
          "витрати на захоплення/звільнення та серіалізація критичної секції.")


def save_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Симуляція банківського рахунку: Lock vs без синхронізації")
    ap.add_argument("--quick", action="store_true", help="скорочений набір експериментів")
    ap.add_argument("--out", default=".", help="каталог для CSV-результатів")
    args = ap.parse_args()

    q = args.quick
    repeats = 3 if q else 5
    out = Path(args.out)

    experiment_basic(threads=8, ops=1000 if q else 2000)
    experiment_gil_note(threads=8, ops=1000 if q else 2000, repeats=repeats)

    scal_rows: list = []
    experiment_scaling("threads", [1, 2, 4, 8, 16, 32] if not q else [1, 4, 16],
                       fixed=500 if q else 1000, repeats=repeats, rows_csv=scal_rows)
    experiment_scaling("ops", [50, 200, 1000, 5000] if not q else [50, 500, 2000],
                       fixed=8, repeats=repeats, rows_csv=scal_rows)

    time_rows: list = []
    experiment_timing([1, 2, 4, 8, 16] if not q else [1, 4, 16],
                      ops=20000 if not q else 3000, repeats=repeats, rows_csv=time_rows)

    save_csv(out / "results_scaling.csv",
             ["vary", "value", "unsafe_wrong_runs", "unsafe_mean_abs_error", "unsafe_negative_events",
              "unsafe_log_breaks", "safe_wrong_runs", "safe_mean_abs_error", "safe_negative_events"],
             scal_rows)
    save_csv(out / "results_timing.csv",
             ["latency", "threads", "total_ops", "unsafe_time_s", "safe_time_s", "ratio"], time_rows)
    print(f"\nCSV збережено: {out / 'results_scaling.csv'}, {out / 'results_timing.csv'}")


if __name__ == "__main__":
    main()
