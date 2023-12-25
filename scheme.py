
import typing
from dataclasses import dataclass, field
from enum import Enum
import math, cmath
import numbers
from fractions import Fraction
import functools
import operator
import itertools
import re



@dataclass
class SchemeList:
	values: tuple["SchemeType", ...]
	proper: bool = True

	def __str__(self):
		if self.proper:
			return "(" + " ".join(map(str, self.values)) + ")"
		elif len(self.values) <= 1: # ?? should not happen
			raise RuntimeError("Encountered a dotted pair with less than two values")
		else:
			return "(" + " ".join(map(str, self.values[:-1])) + " . " + str(self.values[-1]) + ")"

@dataclass
class SchemeSymbol:
	name: str
	def __str__(self):
		return self.name

@dataclass
class SchemeChar:
	value: str
	def __str__(self):
		return "#\\" + self.value

@dataclass
class BuiltinProcedure:
	procId: str
	def __str__(self):
		return f"<proc {self.procId}>"

@dataclass
class DefinedProcedure:
	arguments: tuple[str, ...]
	expr: typing.Union["SchemeType", list["SchemeType"]]
	def __str__(self):
		return f"<proc>"

SchemeType = typing.Union[
	numbers.Number, bool, str, SchemeList, SchemeSymbol, SchemeChar,
	BuiltinProcedure, DefinedProcedure
]



defaultScope: dict[str, SchemeType] = {}
for name in [
	"quote", "define", "set!", "let", "let*", "lambda",
	"apply", "begin", "if", "and", "or",
	"not", "display", "list", "equal?",
	"null?", "pair?", "list?",
	"boolean?", "char?", "symbol?", "string?",  "procedure?", "number?",
	"complex?", "real?", "rational?", "integer?",
	"finite?", "nan?", "zero?", "positive?", "negative?", "odd?", "even?",
	"rationalize",
	"max", "min", "abs",
	"+", "-", "*", "/",
	"<", "<=", "=", ">=", ">",
	"quotient", "remainder", "modulo",
	"floor", "ceiling", "truncate", "round",
	"gcd", "lcm",
	"exp", "log",
	"sin", "cos", "tan", "asin", "acos", "atan",
	"square", "sqrt", "expt",
	"make-rectangular", "make-polar",
	"real-part", "imag-part", "magnitude", "angle",
	"random",
	"cons", "car", "cdr", "length", "reverse", "append",
	"map",
]:
	defaultScope[name] = BuiltinProcedure(name)



def checkArgs(procId, args, minNum=None, maxNum=None, validType=None):
	if minNum is not None:
		if maxNum is None:
			maxNum = minNum
		if not (minNum <= len(args) <= maxNum):
			raise TypeError(f"Wrong number of arguments for '{procId}': {len(args)}")
	if validType and not all(isinstance(arg, validType) for arg in args):
		raise TypeError(f"Non-numeric argument in '{procId}'")

def evalEagerBuiltin(
	procId: str,
	args: tuple[SchemeType, ...],
	scope: dict[str, SchemeType],
) -> typing.Optional[SchemeType]:
	"""
	Makes a single call to one of the hardwired procedures.
	Accepts a tuple of already evaluated arguments (lazy
	"procedures" are handled one level up).
	"""

	if procId == "not":
		checkArgs(procId, args, 1)
		return True if args[0] is False else False

	elif procId == "display":
		checkArgs(procId, args, 1)
		print(args[0], end="")
		return None

	elif procId == "list":
		return SchemeList(args)

	elif procId == "equal?":
		checkArgs(procId, args, 2)
		return args[0] == args[1]

	elif procId == "boolean?":
		checkArgs(procId, args, 1)
		return isinstance(args[0], bool)

	elif procId == "char?":
		checkArgs(procId, args, 1)
		return isinstance(args[0], SchemeChar)

	elif procId == "symbol?":
		checkArgs(procId, args, 1)
		return isinstance(args[0], SchemeSymbol)

	elif procId == "string?":
		checkArgs(procId, args, 1)
		return isinstance(args[0], str)

	elif procId == "procedure?":
		checkArgs(procId, args, 1)
		return isinstance(args[0], BuiltinProcedure) or isinstance(args[0], DefinedProcedure)

	elif procId == "null?":
		checkArgs(procId, args, 1)
		return isinstance(args[0], SchemeList) and len(args[0].values) == 0

	elif procId == "pair?":
		checkArgs(procId, args, 1)
		if not isinstance(args[0], SchemeList):
			return False
		if args[0].proper and len(args[0].values) >= 1:
			return True
		if not args[0].proper and len(args[0].values) >= 2:
			return True
		return False

	elif procId == "list?":
		checkArgs(procId, args, 1)
		return isinstance(args[0], SchemeList)

	elif procId in ("number?", "complex?"):
		checkArgs(procId, args, 1)
		return isinstance(args[0], numbers.Number)

	elif procId == "real?":
		checkArgs(procId, args, 1)
		if isinstance(args[0], numbers.Real):
			return True
		if isinstance(args[0], numbers.Complex) and args[0].imag == 0:
			return True
		return False

	elif procId == "rational?":
		checkArgs(procId, args, 1)
		return isinstance(args[0], numbers.Rational) # ?? 0.1 considered irational

	elif procId == "integer?":
		checkArgs(procId, args, 1)
		if isinstance(args[0], numbers.Integral):
			return True
		if isinstance(args[0], numbers.Real) and int(args[0]) == args[0]:
			return True
		return False

	elif procId == "finite?":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], complex):
			return cmath.isfinite(args[0])
		else:
			return math.isfinite(args[0])

	elif procId == "nan?":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], complex):
			return not cmath.isnan(args[0])
		else:
			return not math.isnan(args[0])

	elif procId == "zero?":
		checkArgs(procId, args, 1, validType=numbers.Number)
		return args[0] == 0

	elif procId == "positive?":
		checkArgs(procId, args, 1, validType=numbers.Real)
		return args[0] > 0

	elif procId == "negative?":
		checkArgs(procId, args, 1, validType=numbers.Real)
		return args[0] < 0

	elif procId == "odd?":
		checkArgs(procId, args, 1, validType=numbers.Integral)
		return (args[0] & 1) == 1

	elif procId == "even?":
		checkArgs(procId, args, 1, validType=numbers.Integral)
		return (args[0] & 1) == 0

	elif procId == "rationalize":
		checkArgs(procId, args, 1, validType=numbers.Real)
		if args[1] <= 0:
			raise ValueError("Precision in 'rationalize' must be positive")
		return Fraction(args[0]).limit_denominator(math.floor(1/args[1]))

	elif procId == "max":
		checkArgs(procId, args, 1, float("inf"), validType=numbers.Real)
		return max(args)

	elif procId == "min":
		checkArgs(procId, args, 1, float("inf"), validType=numbers.Real)
		return min(args)

	elif procId == "abs":
		checkArgs(procId, args, 1, validType=numbers.Real)
		return abs(args[0])

	elif procId == "+":
		checkArgs(procId, args, validType=numbers.Number)
		return functools.reduce(operator.add, args, 0)

	elif procId == "-":
		checkArgs(procId, args, 1, float("inf"), validType=numbers.Number)
		if len(args) == 1:
			return -args[0]
		else:
			return args[0] - sum(args[1:])

	elif procId == "*":
		checkArgs(procId, args, validType=numbers.Number)
		return functools.reduce(operator.mul, args, 1)

	elif procId == "/":
		checkArgs(procId, args, 1, float("inf"), validType=numbers.Number)
		if len(args) == 1:
			if isinstance(args[0], numbers.Integral):
				return Fraction(1, args[0])
			else:
				return 1 / args[0]
		else:
			if isinstance(args[0], numbers.Integral):
				result = Fraction(args[0], 1)
			else:
				result = args[0]
			for arg in args[1:]:
				result /= arg
			return result

	elif procId == "<":
		checkArgs(procId, args, 2, float("inf"), validType=numbers.Real)
		for a, b in zip(args[:-1], args[1:]):
			if not (a < b):
				return False
		return True

	elif procId == "<=":
		checkArgs(procId, args, 2, float("inf"), validType=numbers.Real)
		for a, b in zip(args[:-1], args[1:]):
			if not (a <= b):
				return False
		return True

	elif procId == "=":
		checkArgs(procId, args, 2, float("inf"), validType=numbers.Number)
		for a, b in zip(args[:-1], args[1:]):
			if not (a == b):
				return False
		return True

	elif procId == ">=":
		checkArgs(procId, args, 2, float("inf"), validType=numbers.Real)
		for a, b in zip(args[:-1], args[1:]):
			if not (a >= b):
				return False
		return True

	elif procId == ">":
		checkArgs(procId, args, 2, float("inf"), validType=numbers.Real)
		for a, b in zip(args[:-1], args[1:]):
			if not (a > b):
				return False
		return True

	elif procId == "quotient":
		checkArgs(procId, args, 2, validType=numbers.Integral)
		return sign(args[0]) * sign(args[1]) * (abs(args[0]) // abs(args[1]))

	elif procId == "remainder":
		checkArgs(procId, args, 2, validType=numbers.Integral)
		return sign(args[0]) * sign(args[1]) * (abs(args[0]) % abs(args[1]))

	elif procId == "modulo":
		checkArgs(procId, args, 2, validType=numbers.Integral)
		return args[0] % args[1]

	elif procId == "floor":
		checkArgs(procId, args, 1, validType=numbers.Real)
		return math.floor(args[0])

	elif procId == "ceiling":
		checkArgs(procId, args, 1, validType=numbers.Real)
		return math.ceil(args[0])

	elif procId == "truncate":
		checkArgs(procId, args, 1, validType=numbers.Real)
		return math.trunc(args[0])

	elif procId == "round":
		checkArgs(procId, args, 1, validType=numbers.Real)
		return round(args[0])

	elif procId == "gcd":
		checkArgs(procId, args, validType=numbers.Integral)
		return math.gcd(*args)

	elif procId == "lcm":
		checkArgs(procId, args, validType=numbers.Integral)
		return math.lcm(*args)

	elif procId == "exp":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return math.exp(args[0])
		else:
			return cmath.exp(args[0])

	elif procId == "log":
		checkArgs(procId, args, 1, 2, validType=numbers.Number)
		if len(args) == 1:
			if isinstance(args[0], numbers.Real):
				return math.log(args[0])
			else:
				return cmath.log(args[0])
		else:
			if isinstance(args[0], numbers.Real) and isinstance(args[1], numbers.Real):
				return math.log(args[0], args[1])
			else:
				return cmath.log(args[0], args[1])

	elif procId == "sin":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return math.sin(args[0])
		else:
			return cmath.sin(args[0])

	elif procId == "cos":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return math.cos(args[0])
		else:
			return cmath.cos(args[0])

	elif procId == "tan":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return math.tan(args[0])
		else:
			return cmath.tan(args[0])

	elif procId == "asin":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return math.asin(args[0])
		else:
			return cmath.asin(args[0])

	elif procId == "acos":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return math.acos(args[0])
		else:
			return cmath.acos(args[0])

	elif procId == "atan":
		checkArgs(procId, args, 1, 2, validType=numbers.Number)
		if len(args) == 1:
			if isinstance(args[0], numbers.Real):
				return math.atan(args[0])
			else:
				return cmath.atan(args[0])
		else:
			if isinstance(args[0], numbers.Real) and isinstance(args[1], numbers.Real):
				return math.atan(args[0], args[1])
			else:
				raise TypeError("Two-argument atan not supported for complex numbers")

	elif procId == "square":
		checkArgs(procId, args, 1, validType=numbers.Number)
		return args[0] * args[0]

	elif procId == "sqrt":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return math.sqrt(args[0])
		else:
			return cmath.sqrt(args[0])

	elif procId == "expt":
		checkArgs(procId, args, 2, validType=numbers.Number)
		return args[0] ** args[1]

	elif procId == "make-rectangular":
		checkArgs(procId, args, 2, validType=numbers.Real)
		return complex(args[0], args[1])

	elif procId == "make-polar":
		checkArgs(procId, args, 2, validType=numbers.Real)
		return complex(math.cos(args[1])*args[0], math.sin(args[1])*args[0])

	elif procId == "real-part":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return args[0]
		else:
			return args[0].real

	elif procId == "imag-part":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return args[0] * 0
		else:
			return args[0].imag

	elif procId == "magnitude":
		checkArgs(procId, args, 1, validType=numbers.Number)
		return abs(args[0])

	elif procId == "angle":
		checkArgs(procId, args, 1, validType=numbers.Number)
		if isinstance(args[0], numbers.Real):
			return args[0] * 0
		else:
			return cmag.phase(args[0])

	elif procId == "random":
		checkArgs(procId, args, 0)
		return random.random()

	elif procId == "cons":
		checkArgs(procId, args, 2)
		if isinstance(args[1], SchemeList):
			return SchemeList((args[0], *args[1].values), proper=args[1].proper)
		else:
			return SchemeList((args[0], args[1]), proper=False)

	elif procId == "car":
		checkArgs(procId, args, 1, validType=SchemeList)
		if not args[0].values:
			raise ValueError("'car' on an empty list")
		return args[0].values[0]

	elif procId == "cdr":
		checkArgs(procId, args, 1, validType=SchemeList)
		if not args[0].values:
			raise ValueError("'cdr' on an empty list")
		if len(args[0].values) == 2 and not args[0].proper:
			return args[0].values[1]
		return SchemeList(args[0].values[1:], proper=args[0].proper)

	elif procId == "length":
		checkArgs(procId, args, 1, validType=SchemeList)
		if not args[0].proper:
			raise TypeError("'length' does not work on improper lists")
		return len(args[0].values)

	elif procId == "reverse":
		checkArgs(procId, args, 1, validType=SchemeList)
		if not args[0].proper:
			raise TypeError("'reverse' does not work on improper lists")
		return SchemeList(args[0].values[::-1])

	elif procId == "append":
		checkArgs(procId, args, 1, float("inf"))
		for arg in args[:-1]:
			if not isinstance(arg, SchemeList):
				raise TypeError(f"Appending to a {type(arg).__name__}")
			elif not arg.proper:
				raise TypeError("Appending to an improper list")
		if len(args) == 1:
			return args[0]
		elif not isinstance(args[-1], SchemeList):
			return SchemeList(
				tuple(itertools.chain(*(arg.values for arg in args[:-1]))) + (args[-1], ),
				proper = False,
			)
		else:
			return SchemeList(
				tuple(itertools.chain(*(arg.values for arg in args))),
				proper = args[-1].proper,
			)

	else:
		raise ValueError(f"Unknown built-in procedure {procId}")



def evalProgram(
	program: typing.Union[SchemeType, list[SchemeType]],
	scope: typing.Optional[dict[str, SchemeType]] = None,
) -> typing.Optional[SchemeType]:
	"""
	Recursively evaluates a sequence of Scheme expressions
	"""

	if scope is None:
		scope = defaultScope.copy()

	if isinstance(program, list):
		scope = scope.copy() # avoid propagating definitions upwards
	else:
		program = [program]

	if not program:
		raise ValueError("Evaluating an empty program")

	result: typing.Optional[SchemeType] = None
	for expr in program:

		if isinstance(expr, numbers.Number) or isinstance(expr, bool) \
		or isinstance(expr, str) or isinstance(expr, SchemeChar) \
		or isinstance(expr, BuiltinProcedure) or isinstance(expr, DefinedProcedure):
			result = expr

		elif isinstance(expr, SchemeSymbol):
			if expr.name in scope:
				result = scope[expr.name]
			else:
				raise KeyError(f"Variable '{expr.name}' not defined")

		elif isinstance(expr, SchemeList):
			if len(expr.values) == 0:
				raise ValueError("Evaluating an empty list")

			procedure = evalProgram(expr.values[0], scope)

			if isinstance(procedure, BuiltinProcedure):
				procId = procedure.procId
				args = expr.values[1:]

				# builtin forms (lazy)

				if procId == "quote":
					if len(args) != 1:
						raise TypeError(f"Applying 'quote' to {len(args)} values (expected 1)")
					result = args[0]

				elif procId in ("define", "set!"):
					if len(args) != 2:
						raise TypeError(f"A '{procId}' with {len(args)} arguments (expected 2)")
					if not isinstance(args[0], SchemeSymbol):
						raise TypeError(f"Invalid type of identifier in '{procId}': {type(args[0]).__name__}")
					name = args[0].name
					if procId == "define" and name in scope:
						raise RuntimeError(f"Redefining variable '{name}'")
					scope[name] = evalProgram(args[1], scope)
					result = None

				elif procId in ("let", "let*"):
					if len(args) < 2:
						raise TypeError(f"A '{procId}' with {len(args)} arguments (expected at least 2)")
					if not isinstance(args[0], SchemeList):
						raise TypeError(f"Expected a list of assignment pairs in '{procId}'")
					exprScope = scope.copy()
					assignmentScope = exprScope if procId == "let*" else {}
					for pair in args[0].values:
						if not isinstance(pair, SchemeList):
							raise TypeError(f"Expected an assignment pair in '{procId}', got {type(pair).__name__}")
						if len(pair.values) != 2:
							raise TypeError(f"Assignment pair with {len(pair)} items in '{procId}'")
						symbol, value = pair.values
						if not isinstance(symbol, SchemeSymbol):
							raise TypeError(f"Invalid type of identifier in '{procId}': {type(symbol).__name__}")
						assignmentScope[symbol.name] = evalProgram(value, exprScope)
					if assignmentScope is not exprScope:
						exprScope.update(assignmentScope)
					if len(args) == 2:
						result = evalProgram(args[1], exprScope)
					else:
						result = evalProgram(list(args[1:]), exprScope)

				elif procId == "lambda":
					if len(args) < 2:
						raise TypeError(f"A '{procId}' with {len(args)} arguments (expected at least 2)")
					if not isinstance(args[0], SchemeList):
						raise TypeError(f"Expected a list as the first argument of '{procId}'")
					if not all(isinstance(arg, SchemeSymbol) for arg in args[0].values):
						raise TypeError(f"A '{procId}' argument which is not a symbol")
					procArgs = tuple(arg.name for arg in args[0].values)
					if len(args) == 2:
						result = DefinedProcedure(procArgs, args[1])
					else:
						result = DefinedProcedure(procArgs, list(args[1:]))

				elif procId == "apply":
					if len(args) < 2:
						raise TypeError(f"A '{procId}' with {len(args)} arguments (expected at least 2)")
					if not isinstance(args[-1], SchemeList):
						raise TypeError(f"The last argument of '{procId}' must be a list, got {type(args[-1]).__name__}")
					last = evalProgram(args[-1], scope)
					result = evalProgram(SchemeList(args[:-1] + last.values), scope)

				elif procId == "map":
					if len(args) < 2:
						raise TypeError(f"A '{procId}' with {len(args)} arguments (expected at least 2)")
#					args = tuple(evalProgram(arg, scope) for arg in args)
					argValues = []
					for arg in args:
						argValues.append(evalProgram(arg, scope))
					if not isinstance(argValues[0], BuiltinProcedure | DefinedProcedure):
						raise TypeError(f"'map' called with a {type(argValues[0]).__name__} as procedure")
					lengths = set()
					for arg in argValues[1:]:
						if not isinstance(arg, SchemeList):
							raise TypeError(f"'map' called with a {type(arg).__name__} as list")
						if not arg.proper:
							raise TypeError(f"'map' does not work with improper lists")
						lengths.add(len(arg.values))
					if len(lengths) > 1:
						raise ValueError("Lists passed to 'map' differ in length")
					result = []
					for i in range(lengths.pop()):
						mapExpr = SchemeList((argValues[0], *(arg.values[i] for arg in argValues[1:])))
						result.append(evalProgram(mapExpr, scope))
					result = SchemeList(tuple(result))

				elif procId == "begin":
					result = evalProgram(list(args))

				elif procId == "if":
					if len(args) != 3:
						raise TypeError(f"An '{procId}' with {len(args)} arguments (expected 3)")
					condValue = evalProgram(args[0], scope)
					if condValue is not False:
						result = evalProgram(args[1], scope)
					else:
						result = evalProgram(args[2], scope)

				elif procId == "and":
					result = True
					for arg in args:
						value = evalProgram(arg, scope)
						if value is False:
							result = False
							break
						else:
							result = value

				elif procId == "or":
					result = False
					for arg in args:
						value = evalProgram(arg, scope)
						if value is not False:
							result = value
							break

				# builtin procedures (eager)

				else:
#					args = tuple(evalProgram(arg, scope) for arg in args)
					argValues = []
					for arg in args:
						argValues.append(evalProgram(arg, scope))
					result = evalEagerBuiltin(procId, tuple(argValues), scope)

			# defined procedures (eager)

			elif isinstance(procedure, DefinedProcedure):

#				args = tuple(evalProgram(arg, scope) for arg in expr.values[1:])
				argValues = []
				for arg in expr.values[1:]:
					argValues.append(evalProgram(arg, scope))
				subScope = scope.copy()
				assert len(procedure.arguments) == len(argValues)
				for name, value in zip(procedure.arguments, argValues): #, strict=True):
					subScope[name] = value
				result = evalProgram(procedure.expr, subScope)

			else:
				raise TypeError(f"Cannot evaluate a {type(procedure).__name__} as a procedure")

		else:
			raise TypeError(f"Cannot evaluate a {type(expr).__name__} as a Scheme expression")

	return result



@dataclass
class ASTScope:
	quoted: bool = False
	items: list[SchemeType] = field(default_factory=list)

def parse(code: str):
	"""
	Not complete, not fully debugged.
	Used only for easier test input creation.
	"""

	integerPattern = r"[+\-]?\d+"
	rationalPattern = r"[+\-]?\d+/\d+"
	realPattern = r"[+\-]?\d+ (?:\.\d*)? (?:[eE][+\-]?\d+)?"
	complexPattern = "{real}[+\-]{real}i".format(real=realPattern)
	endPattern = r"(?= \s | $ | \))"

	integerRe = re.compile(integerPattern+endPattern, re.VERBOSE)
	rationalRe = re.compile(rationalPattern+endPattern, re.VERBOSE)
	realRe = re.compile(realPattern+endPattern, re.VERBOSE)
	complexRe = re.compile(complexPattern+endPattern, re.VERBOSE)

	# tokenization
	tokenRe = re.compile(r"""
	  \(                             # list start
	| '\(                            # quoted list start
	| \)                      {end}  # list end
	| \.                      {end}  # short dotted pair syntax
	| \#[tf]                  {end}  # boolean literals
	| \#\\\S+                 {end}  # char literal
	| {rational}              {end}  # fraction
	| {real}                  {end}  # integer or real number
	| {complex}               {end}  # complex number
	| " (?: \\\\" | [^"] )* " {end}  # string
	| [^\s()'.#\\]+           {end}  # symbol
	""".format(
		rational = rationalPattern,
		real = realPattern,
		complex = complexPattern,
		end = endPattern,
	), re.VERBOSE)
	tokens = tokenRe.findall(code)

	if re.sub(r"\s", "", "".join(tokens)) != re.sub(r"\s", "", code):
		raise SyntaxError("Something was left out in Scheme tokenization")

	# treeifying
	stack = [ASTScope()]
	for token in tokens:
		token = token.strip()
		if token == "(":
			stack.append(ASTScope())
		elif token == "'(":
			stack.append(ASTScope(quoted=True))
		elif token == ")":
			record = stack.pop()
			item = SchemeList(tuple(record.items))
			if record.quoted:
				item = SchemeList((SchemeSymbol("quote"), item)) # ?? fails when quote is redefined
			if stack:
				stack[-1].items.append(item)
			else:
				raise SyntaxError("Too many ')' in Scheme code")
		elif token == "#t":
			stack[-1].items.append(True)
		elif token == "#f":
			stack[-1].items.append(False)
		elif token.startswith("#\\"):
			stack[-1].items.append(SchemeChar(token[2:]))
		elif rationalRe.match(token):
			stack[-1].items.append(Fraction(token))
		elif integerRe.match(token):
			stack[-1].items.append(int(token))
		elif realRe.match(token):
			stack[-1].items.append(float(token))
		elif complexRe.match(token):
			stack[-1].items.append(complex(token.replace("i", "j")))
		elif token[0] == "\"" and token[-1] == "\"" and len(token) >= 2:
			stack[-1].items.append(token[1:-1])
		else: # symbol?
			stack[-1].items.append(SchemeSymbol(token))

	if len(stack) > 1:
		raise SyntaxError("Not enough ')' in Scheme code")

	return stack[0].items



