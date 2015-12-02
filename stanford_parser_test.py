import os
import re

sentence = "I made this for thanksgiving, everyone loved it! So easy to make. I used a chocolate graham cracker crust just to make it even better."
os.popen("echo '"+sentence+"' > ~/stanfordtemp.txt")
parser_out = os.popen("~/stanford-parser-2012-11-12/lexparser.sh ~/stanfordtemp.txt").readlines()
bracketed_parse = " ".join( [i.strip() for i in parser_out if len(i.strip()) > 0 and i.strip()[0] == "("])

print bracketed_parse

class Node(object):
	def __init__(self, data):
		self.data = data
		self.left = None
		self.right = None
		self.parent = None

# Split the parse into an array
split =  re.split('(\W)', bracketed_parse)

# We will store the leaf nodes here in the correct order
node_list = []

def add_child(parent, symbol, leaf_status):
	child = Node(symbol)
	child.parent = parent
	if (parent):
		if (parent.left==None):
			parent.left = child
		else:
			parent.right = child
	else: 
		root = child
	if (leaf_status):
		node_list.append(child)
	return child

# return true if symbol is not a parenthesis or separator
def is_tag(symbol): 
	if (symbol in [")", "("] or str(symbol).isspace() or str(symbol)==""):
		return False
	return True

current_parent = None
current_node = None
root = None
last_symbol = None # Store the last symbol we've seen to identify the leaf nodes

# Go through the array and build the tree
# The leaf node's type is stored as its PARENT
for symbol in split:
	if (str(symbol).isspace() or str(symbol)==""):
		pass
	else:
		if (symbol=="("):
			if (last_symbol!=")"):	
				current_parent = current_node
		elif (symbol==")"):
			current_parent = current_parent.parent
		elif (is_tag(last_symbol)): # Leaf node case
			current_parent = current_node
			child = add_child(current_parent, symbol, True)
			current_node = child
		else:
			child = add_child(current_parent, symbol, False)
			current_node = child
		last_symbol = symbol


# Define segments according to rules
def is_coordinating_conjunction(word):
	if (word.parent.data == "CC"):
		if (word.parent.parent and word.parent.parent.data not in ["NP, PP, ADJP"]):
			return True
	return False

def is_comma_condition(word):
	if (word.data=="," and word.parent.parent and word.parent.parent.data not in ["NP, PP, ADJP"]):
		return True

def is_break_symbol(word):
	if (word.data in [".", ":", "!"]):
		return True
	return False


# Now just segment based on the rules and print the segments
current_segment = []
for i in range(len(node_list)):
	word= node_list[i]
	current_segment.append(word.data)

	# Hack to get the spacing correct before punctuation
	if (is_break_symbol(word) or is_comma_condition(word)):
		joined_str = ' '.join(current_segment[:-1])
		print joined_str + current_segment[-1]
		current_segment = []
	if (is_coordinating_conjunction(word)): #sentence break
		print ' '.join(current_segment)
		current_segment = []
print ' '.join(current_segment)
