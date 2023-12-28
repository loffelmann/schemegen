
import unittest
from unittest.mock import patch, call
from fractions import Fraction

from scheme import *



class EvalTestCase(unittest.TestCase):

	def findDifference(self, a, b):
		assert isinstance(a, SchemeType | type(None)), f"Cannot compare {type(a)}"
		if type(a) is not type(b):
			return(f"type mismatch: {type(a)} vs {type(b)}")
		if isinstance(a, SchemeList):
			if len(a.values) != len(b.values):
				return(f"different length of Scheme lists: {len(a.values)} vs {len(b.values)}")
			if a.proper != b.proper:
				return(f"mismatching list proper-ness")
			for subA, subB in zip(a.values, b.values):
				subError = self.findDifference(subA, subB)
				if subError:
					return subError
			return None
		else:
			if a != b:
				return(f"value mismatch: {a} vs {b}")
			return None

	def assertSchemeEqual(self, a, b, code):
		dif = self.findDifference(a, b)
		self.assertIsNone(dif, f"{dif}\ncode: {code}")

	def eval(self, code):
		program = parse(code)
		return evalProgram(program)

	@patch("builtins.print")
	def check(self, code, expectedResult, mockPrint=None, *, prints=None):
		program = parse(code)
		result = evalProgram(program)
		self.assertSchemeEqual(result, expectedResult, code)
		if mockPrint is not None and prints is not None:
			self.assertEqual(prints, mockPrint.mock_calls, "mismatching print calls")

	def test_equalityTest(self):
		self.assertIsNone(self.findDifference(1, 1))
		self.assertIsNone(self.findDifference(2.0, 2.0))
		self.assertIsNotNone(self.findDifference(1, 2))
		self.assertIsNotNone(self.findDifference(1, 1.0))
		self.assertIsNotNone(self.findDifference(1.0, complex(1, 0)))
		self.assertIsNotNone(self.findDifference(1, None))
		self.assertIsNotNone(self.findDifference(None, 1))
		self.assertIsNone(self.findDifference("hello", "hello"))
		self.assertIsNotNone(self.findDifference("hello", "Hello"))
		self.assertIsNotNone(self.findDifference("hello", SchemeSymbol("hello")))
		self.assertIsNotNone(self.findDifference("5", 5))
		self.assertIsNone(self.findDifference(
			SchemeList((1, 2, 3)),
			SchemeList((1, 2, 3)),
		))
		self.assertIsNotNone(self.findDifference(
			SchemeList((1, 2, 3)),
			SchemeList((1, 2, 3, 4)),
		))
		self.assertIsNotNone(self.findDifference(
			SchemeList((1, 2, 3)),
			SchemeList((1, 2, 3), proper=False),
		))
		self.assertIsNone(self.findDifference(
			SchemeList((1, 2, 3, SchemeList((4, 5)))),
			SchemeList((1, 2, 3, SchemeList((4, 5)))),
		))
		self.assertIsNotNone(self.findDifference(
			SchemeList((1, 2, 3, SchemeList((4, 5)))),
			SchemeList((1, 2, 3, SchemeList((4.0, 5)))),
		))



class BuiltinFunctionTest(EvalTestCase):

	def test_listConstruction(self):
		self.check("'()", SchemeList(()))
		self.check("'(1)", SchemeList((1, )))
		self.check("'(1 2 3)", SchemeList((1, 2, 3)))
		self.check("'(1 (+ 1 2))", SchemeList((1, SchemeList((SchemeSymbol("+"), 1, 2)))))

		self.check("(quote ())", SchemeList(()))
		self.check("(quote (1))", SchemeList((1, )))
		self.check("(quote (1 2 3))", SchemeList((1, 2, 3)))
		self.check("(quote (1 (+ 1 2)))", SchemeList((1, SchemeList((SchemeSymbol("+"), 1, 2)))))

		self.check("(list )", SchemeList(()))
		self.check("(list 1)", SchemeList((1, )))
		self.check("(list 1 2 3)", SchemeList((1, 2, 3)))
		self.check("(list 1 (+ 1 2))", SchemeList((1, 3)))

		self.check("(cons 1 2)", SchemeList((1, 2), proper=False))
		self.check("(cons 1 '())", SchemeList((1, )))

		self.check("(append '(1 2) '(3 4))", SchemeList((1, 2, 3, 4)))
		self.check("(append '(1 2) '(3 4) 5)", SchemeList((1, 2, 3, 4, 5), proper=False))

	def test_listManipulation(self):
		self.check("(car (list 1 2 3))", 1)
		self.check("(cdr (list 1 2 3))", SchemeList((2, 3)))
		self.check("(car (cdr (list 1 2 3)))", 2)
		self.check("(cdr (car (list (list 1 2 3) 4 5)))", SchemeList((2, 3)))
		self.check("(car (cons 1 2))", 1)
		self.check("(cdr (cons 1 2))", 2)
		self.check("(length (list 1 2 3))", 3)
		self.check("(reverse (list 1 2 3))", SchemeList((3, 2, 1)))

	def test_arith(self):
		self.check("(+)", 0)
		self.check("(+ 5)", 5)
		self.check("(+ -5)", -5)
		self.check("(+ 1 2)", 3)
		self.check("(+ -3 4)", 1)
		self.check("(+ 1 2 3 4)", 10)

		with self.assertRaises(TypeError):
			self.eval("(-)")
		self.check("(- 3)", -3)
		self.check("(- 1 2)", -1)
		self.check("(- -3 4)", -7)
		self.check("(- 10 1 1 1)", 7)

		self.check("(*)", 1)
		self.check("(* 2)", 2)
		self.check("(* 1 2)", 2)
		self.check("(* -3 4)", -12)
		self.check("(* -2 3 -4)", 24)

		with self.assertRaises(TypeError):
			self.eval("(/)")
		self.check("(/ 10)", Fraction(1, 10))
		self.check("(/ 1 2)", Fraction(1, 2))
		self.check("(/ -3 4)", Fraction(-3, 4))
		with self.assertRaises(ZeroDivisionError):
			self.eval("(/ 1 0)")

		self.check("(+ (* 2 3) (/ 10 20) (- (* 3 5) 5))", Fraction(33, 2))

	def test_arithCoercion(self):
		self.check("(+ 1 1/2)", Fraction(3, 2))
		self.check("(+ 1 2.0)", 3.0)
		self.check("(+ 1.0 2)", 3.0)
		self.check("(+ 1.0 2+1i)", 3.0+1j)
		self.check("(+ 1/2 2-1i)", 2.5-1j)
		self.check("(+ 2+1i 2-1i)", complex(4, 0))

		self.check("(- 1 1/2)", Fraction(1, 2))
		self.check("(- 1 2.0)", -1.0)
		self.check("(- 1.0 2)", -1.0)
		self.check("(- 1.0 2+1i)", complex(-1, -1))
		self.check("(- 1/2 2-1i)", complex(-1.5, 1))

		self.check("(* 1 1/2)", Fraction(1, 2))
		self.check("(* 1 2.0)", 2.0)
		self.check("(* 1.0 2)", 2.0)
		self.check("(* 1.0 2+1i)", 2.0+1j)
		self.check("(* 1/2 2-1i)", 1-0.5j)

		self.check("(/ 1 1/2)", Fraction(2, 1))
		self.check("(/ 1 2.0)", 0.5)
		self.check("(/ 1.0 2)", 0.5)
		self.check("(/ 1.0 2+1i)", 1/complex(2, 1))
		self.check("(/ 1/2 2-1i)", 0.5/complex(2, -1))

	def test_variable(self):
		self.check("(define a 1) (define b 2) (list a b)", SchemeList((1, 2)))
		with self.assertRaises(RuntimeError):
			self.eval("(define a 1) (define a 2)")
		self.check("(define a 1) (set! a 2) a", 2)
		self.check("(set! a 1) (set! a 2) a", 2)
		self.check("(define a 1) (set! a 2) (set! a 3) a", 3)
		with self.assertRaises(RuntimeError):
			self.eval("(set! a 1) (define a 2)")

		self.check("(let ((a 2)) (* a a))", 4)
		self.check("(let ((a 2) (b 3)) (- b a))", 1)
		self.check("(let ((a 2)) (set! a 3) (* a a))", 9)
		with self.assertRaises(KeyError):
			self.eval("(let ((a 2)) a) a")
		with self.assertRaises(KeyError):
			self.eval("(let ((a 2) (b (* 3 a))) (- b a))")
		self.eval("(let* ((a 2) (b (* 3 a))) (- b a))")

	def test_logic(self):
		self.check("(if #t 1 2)", 1)
		self.check("(if #f 1 2)", 2)
		self.check("(if 0 2 3)", 2)
		self.check("(if '() 3 4)", 3)

		self.check("(and)", True)
		self.check("(and #t)", True)
		self.check("(and #f)", False)
		self.check("(and #t #t)", True)
		self.check("(and #t #f)", False)
		self.check("(and #f #t)", False)
		self.check("(and #f #f)", False)
		self.check("(and #t #t #t)", True)
		self.check("(and #t #t #t #f)", False)
		self.check("(and #t '())", SchemeList(()))
		self.check("(and #t 0)", 0)
		self.check("(and '() #t)", True)
		self.check("(and 0 #t)", True)
		self.check("(and #t '(1 2 3))", SchemeList((1, 2, 3)))

		self.check("(or)", False)
		self.check("(or #t)", True)
		self.check("(or #f)", False)
		self.check("(or #t #t)", True)
		self.check("(or #t #f)", True)
		self.check("(or #f #t)", True)
		self.check("(or #f #f)", False)
		self.check("(or #f #f #f)", False)
		self.check("(or #f #f #f #t)", True)
		self.check("(or #t '())", True)
		self.check("(or #f '())", SchemeList(()))
		self.check("(or #t 0)", True)
		self.check("(or '() #t)", SchemeList(()))
		self.check("(or 0 #t)", 0)
		self.check("(or #f '(1 2 3))", SchemeList((1, 2, 3)))

		with self.assertRaises(TypeError):
			self.eval("(not)")
		self.check("(not #t)", False)
		self.check("(not #f)", True)
		self.check("(not 0)", False)
		self.check("(not '())", False)
		self.check("(not (list 1 2 3))", False)
		with self.assertRaises(TypeError):
			self.eval("(not #t #f)")

	def test_lazy(self):
		self.check("(if #t 1 (/ 1 0))", 1)
		self.check("(if #f (/ 1 0) 2)", 2)
		with self.assertRaises(ZeroDivisionError):
			self.eval("(if #f 3 (/ 1 0))")
		with self.assertRaises(ZeroDivisionError):
			self.eval("(if #t (/ 1 0) 4)")

		self.check("(and #f (/ 1 0))", False)
		self.check("(and 0 #f (/ 1 0))", False)
		self.check("(and #t #f #t (/ 1 0))", False)
		with self.assertRaises(ZeroDivisionError):
			self.eval("(and #t (/ 1 0))")
		with self.assertRaises(ZeroDivisionError):
			self.eval("(and #t 0 '() (/ 1 0))")

		self.check("(or #t (/ 1 0))", True)
		self.check("(or #f 0 (/ 1 0))", 0)
		self.check("(or #f #t #f (/ 1 0))", True)
		with self.assertRaises(ZeroDivisionError):
			self.eval("(or #f (/ 1 0))")
		with self.assertRaises(ZeroDivisionError):
			self.eval("(or #f #f (/ 1 0) #t #f)")

	def test_misc(self):
		self.check("(display 1)", None, prints=[call(1, end="")])
		self.check("(display \"hello world\")", None, prints=[call("hello world", end="")])



class ProcedureTest(EvalTestCase):

	def test_procedure(self):
		self.check("((lambda (x) (* x x x)) 5)", 125)

		self.check("""
		(define cube (lambda (x) (* x x x)))
		(cube 3)
		""", 27)
		self.check("""
		(define helper (lambda (x) (* x x)))
		(define cube (lambda (x) (* x (helper x))))
		(cube 2)
		""", 8)

		self.check("(apply * (list 4 5))", 20)
		self.check("(apply * 3 (list 4 5))", 60)
		self.check("""
		(define x 10)
		(apply * x (list 4 5))
		""", 200)
		self.check("""
		(define mul3 (lambda (a b c) (* a b c)))
		(apply mul3 6 (list 4 5))
		""", 120)

		self.check("(map square (list 1 2 3))", SchemeList((1, 4, 9)))
		self.check("(map * (list 1 2 3) (list 4 5 6))", SchemeList((4, 10, 18)))
		self.check("""
		(define mul3 (lambda (a b c) (* a b c)))
		(map mul3 (list 4 5) (list 10 20) (list 1 2))
		""", SchemeList((40, 200)))

		with self.assertRaises(TypeError):
			self.eval("""
				(define proc (lambda (a b) (+ a b)))
				(proc 1)
			""")
		with self.assertRaises(TypeError):
			self.eval("""
				(define proc (lambda (a b) (+ a b)))
				(proc 1 2 3)
			""")

	def test_sequence(self):
		self.check("""
		(begin
			(display "start")
			(define x 5)
			(define y (* x x))
			(display y)
			(+ x y)
		)
		""", 30, prints=[
			call("start", end=""),
			call(25, end=""),
		])

		self.check("""
		(begin
			(display "level 1")
			(begin
				(display "level 2")
				(begin
					(display "level 3")
				)
				(display "level 2 again")
			)
			(display "level 1 again")
		)
		""", None, prints=[
			call("level 1", end=""),
			call("level 2", end=""),
			call("level 3", end=""),
			call("level 2 again", end=""),
			call("level 1 again", end=""),
		])

		self.check("""
		(let
			((a 1) (b 2))
			(display a)
			(display b)
			(+ a b)
		)
		""", 3, prints=[
			call(1, end=""),
			call(2, end=""),
		])

		self.check("""
		(define printAndSum 
			(lambda (a b)
				(display a)
				(display "+")
				(display b)
				(+ a b)
			)
		)
		(printAndSum 7 15)
		""", 22, prints=[
			call(7, end=""),
			call("+", end=""),
			call(15, end=""),
		])

	def test_tailCall(self):
		code = """
		(define fun (lambda (n)
			(if (> n 0)
				(fun (- n 1))
				42
			)
		))
		(fun 10000)
		"""
		self.check(code, 42)

		code = """
		(define fun (lambda (n)
			(if (<= n 0)
				42
				(let ((m (- n 1))) (fun m))
			)
		))
		(fun 10000)
		"""
		self.check(code, 42)

#		# ?? does not work, scoping is wrong
#		code = """
#		(define fun (lambda (n)
#			(if (> n 0)
#				(let ((m (- n 1))) (define o m) (fun o))
#				42
#			)
#		))
#		(fun 10000)
#		"""
#		self.check(code, 42)

		code = """
		(define fun (lambda (n)
			(if (> n 0)
				(apply fun (list (- n 1)))
				42
			)
		))
		(fun 10000)
		"""
		self.check(code, 42)

		code = """
		(define fun (lambda (n)
			(if (> n 0)
				(begin
					(display "hi")
					(fun (- n 1))
				)
				42
			)
		))
		(fun 10000)
		"""
		self.check(code, 42)

		code = """
		(define fun (lambda (n)
			(if (> n 0)
				(and 1 (fun (- n 1)))
				42
			)
		))
		(fun 10000)
		"""
		self.check(code, 42)

		code = """
		(define fun (lambda (n)
			(if (> n 0)
				(or #f #f #f (fun (- n 1)))
				42
			)
		))
		(fun 10000)
		"""
		self.check(code, 42)

	def test_integration(self):
		code = """
		(define fib (lambda (n)
			(if (>= n 2)
				(+ (fib (- n 1)) (fib (- n 2)))
				1
			)
		))
		(fib 10)
		"""
		self.check(code, 89)

		code = """
		(define smallest (lambda (list sofar)
			(if (null? list)
				sofar
				(if (< (car list) sofar)
					(smallest (cdr list) (car list))
					(smallest (cdr list) sofar)
				)
			)
		))
		(smallest (list 10 5 9 4 8 3 7 2 6 1) 10000000)
		"""
		self.check(code, 1)

		code = """
		(define range (lambda (n) (range_helper n '())))
		(define range_helper (lambda (n list)
			(if (> n 0)
				(range_helper (- n 1) (cons (- n 1) list))
				list
			)
		))
		(range 10)
		"""
		self.check(code, SchemeList(tuple(range(10))))



if __name__ == "__main__":
	unittest.main()

