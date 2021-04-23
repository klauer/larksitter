import json
import lark
from pathlib import Path
import pprint
import dataclasses
import marshmallow
from marshmallow import Schema
from marshmallow_dataclass import dataclass
from dataclasses import field
from typing import List, Optional, Dict, Union


@dataclass
class ChildQuantity:
    exists: bool
    required: bool
    multiple: bool


@dataclass
class FieldInfo:
    # quantity: ChildQuantity
    multiple: bool
    required: bool
    types: List["ChildType"]

    def as_lark(self):
        if self.multiple:
            if self.required:
                start, end = "(", ")+"
            else:
                start, end = "(", ")*"
        else:
            if self.required:
                start, end = "(", ")"
            else:
                start, end = "[", "]"

        child_types = " | ".join(child_type.as_lark() for child_type in self.types)
        return f"{start}{child_types}{end}"


@dataclass
class ChildType:
    named: bool
    type: str

    def as_lark(self):
        if self.named:
            return self.type
        return f'"{self.type}"'


@dataclass
class NodeType:
    type: str
    named: bool

    def as_lark(self):
        if self.named:
            return self.type
        return f'"{self.type}"'


@dataclass
class NodeInfo:
    type: str
    named: bool
    fields: Optional[Dict[str, FieldInfo]] = field(default_factory=dict)
    children: Optional[FieldInfo] = field(default=None)
    subtypes: Optional[List["NodeType"]] = field(default_factory=list)

    def as_lark(self):
        if not self.named:
            return f"// unnamed node {self.type!r}"

        if self.subtypes:
            assert not self.children
            param_list = [param.as_lark() for param in self.subtypes]
        elif self.children:
            assert not self.subtypes
            param_list = [self.children.as_lark()]
        elif self.fields:
            param_list = [
                f"{name}={field.as_lark()}" for name, field in self.fields.items()
            ]
        else:
            param_list = []

        if not param_list:
            return f"// {self.type}: "

        param_list = " | ".join(param_list)
        return f"// {self.type}: {param_list}"


# ^ nodes above, grammar below


@dataclass
class GrammarRule:
    def __post_init__(self):
        content = getattr(self, "content", {})
        if content:
            self.content = GrammarRule.from_dict(content)

        members = getattr(self, "members", {})
        if members:
            self.members = [GrammarRule.from_dict(member) for member in self.members]

    @classmethod
    def from_dict(cls, info):
        info = dict(info)
        type_ = info.pop("type")
        cls = {
            "ALIAS": Alias,
            "BLANK": Blank,
            "CHOICE": Choice,
            "FIELD": Field,
            "IMMEDIATE_TOKEN": ImmediateToken,
            "PATTERN": Pattern,
            "PREC_DYNAMIC": PrecDynamic,
            "PREC_LEFT": PrecLeft,
            "PREC_RIGHT": PrecRight,
            "PREC": Prec,
            "REPEAT1": Repeat1,
            "REPEAT": Repeat,
            "SEQ": Seq,
            "STRING": String,
            "SYMBOL": Symbol,
            "TOKEN": Token,
        }[type_]
        return cls.Schema().load(info)


@dataclass
class Alias(GrammarRule):
    content: dict
    named: bool
    value: str

    def as_lark(self):
        return self.content.as_lark()
        # ?
        if self.named:
            return "(alias){self.value}"

        return self.value


@dataclass
class Blank(GrammarRule):
    def as_lark(self):
        return "BLANK"


@dataclass
class String(GrammarRule):
    value: str

    def as_lark(self):
        value = self.value
        if r"\\" not in value:
            value = value.replace("\\", r"\\")
        value = value.replace('"', '\\"')

        value = value.replace('\t', '\\t')
        value = value.replace('\n', '\\n')
        value = value.replace('\r', '\\r')
        return f'"{value}"'


@dataclass
class Pattern(GrammarRule):
    value: str

    def as_lark(self):
        value = self.value
        value = value.replace("/", "\\/")
        value = value.replace('\\"', '"')
        value = value.replace('\t', '\\t')
        value = value.replace('\n', '\\n')
        value = value.replace('\r', '\\r')
        return f"/{value}/"


@dataclass
class Symbol(GrammarRule):
    name: str

    def as_lark(self):
        return self.name


@dataclass
class Choice(GrammarRule):
    members: List[dict]

    def as_lark(self):
        members = [member.as_lark() for member in self.members
                   if not isinstance(member, Blank)]
        if any(isinstance(member, Blank) for member in self.members):
            options = " | ".join(members)
            return f"[{options}]"

        options = " | ".join(members)
        return f"({options})"


@dataclass
class Field(GrammarRule):
    name: str
    content: dict

    def as_lark(self):
        # TODO: name
        return self.content.as_lark()


@dataclass
class Seq(GrammarRule):
    members: List[dict]

    def as_lark(self):
        seq = " ".join(member.as_lark() for member in self.members)
        return f"({seq})"


@dataclass
class Repeat(GrammarRule):
    content: dict

    def as_lark(self):
        return f"({self.content.as_lark()})*"


@dataclass
class Repeat1(GrammarRule):
    content: dict

    def as_lark(self):
        return f"[{self.content.as_lark()}]"


@dataclass
class PrecDynamic(GrammarRule):
    value: int
    content: dict

    def as_lark(self):
        # return "// prec_dynamic?"
        return self.content.as_lark()


@dataclass
class PrecLeft(GrammarRule):
    value: int  # PrecedenceValue
    content: dict

    def as_lark(self):
        # return "// prec_dynamic?"
        return self.content.as_lark()


@dataclass
class PrecRight(GrammarRule):
    value: int  # PrecedenceValue
    content: dict

    def as_lark(self):
        # return "// prec_dynamic?"
        return self.content.as_lark()


@dataclass
class Prec(GrammarRule):
    value: int  # PrecedenceValue
    content: dict

    def as_lark(self):
        return self.content.as_lark()
        # return "// prec_dynamic?"


@dataclass
class Token(GrammarRule):
    content: dict

    def as_lark(self):
        return self.content.as_lark()


@dataclass
class ImmediateToken(GrammarRule):
    content: dict

    def as_lark(self):
        return self.content.as_lark()


def reformat(string, indent):
    return string


@dataclass
class Grammar:
    name: str
    rules: Dict[str, dict]
    precedences: List[List[dict]]
    conflicts: List[List[str]]
    externals: List[dict]
    extras: List[dict]
    inline: List[str]
    supertypes: List[str]
    word: Optional[str]
    PREC: Optional[dict]

    def __post_init__(self):
        for name, rule in list(self.rules.items()):
            self.rules[name] = GrammarRule.from_dict(rule)

    def as_lark(self):
        first_rule = list(self.rules)[0]
        res = [
            f"start: {first_rule}"
        ]
        for rule_name, rule in self.rules.items():
            rule_lark = rule.as_lark()
            if "|" in rule_lark:
                rule_lark = reformat(rule_lark, indent=len(rule_name))
            res.append(f"{rule_name}: {rule_lark}")

        return "\n".join(res)


def reformat_grammar(grammar):
    parser = lark.Lark.open(
        "/Users/klauer/Repos/lark/lark/grammars/lark.lark",
        rel_to=__file__,
        parser="lalr"
    )

    @lark.visitors.v_args(inline=True)
    class GrammarTransformer(lark.Transformer):
        def rule(self, rule, rule_params, *remainder):
            if len(remainder) == 2:
                priority, expansions = remainder
            else:
                priority, expansions = None, remainder

            print("RULE", rule, rule_params, '**', expansions, priority)
            return locals()

    tree = parser.parse(grammar)
    return GrammarTransformer(visit_tokens=True).transform(tree)



def json_test():
    with open("json-grammar.json") as fp:
        grammar = json.load(fp)

    with open("json-node-types.json", "rb") as fp:
        types = json.load(fp)


    sch: Schema = NodeInfo.Schema()
    types = sch.load(types, many=True)
    for typ in types:
        print(typ.as_lark())

    sch: Schema = Grammar.Schema()
    grammar = sch.load(grammar)
    lark_grammar = grammar.as_lark()

    print(lark_grammar)

    parser = lark.Lark(
        lark_grammar,
        parser="lalr",
        # parser="earley",
        # parser="cyk",
        # lexer_callbacks={"COMMENT": comments.append},
        # ambiguity='explicit',
        # lexer='standard',
        # lexer="dynamic_complete",
    )
    print(parser.parse('{"abc":"def"}').pretty())


with open("c-grammar.json") as fp:
    grammar = json.load(fp)

with open("c-nodes.json", "rb") as fp:
    types = json.load(fp)


sch: Schema = NodeInfo.Schema()
types = sch.load(types, many=True)
for typ in types:
    print(typ.as_lark())

sch: Schema = Grammar.Schema()
grammar = sch.load(grammar)
lark_grammar = grammar.as_lark()

print(lark_grammar)

lark_grammar = r"""
start: translation_unit
translation_unit: (_top_level_item)*
_top_level_item: (function_definition | linkage_specification | declaration | _statement | type_definition | _empty_declaration | preproc_if | preproc_ifdef | preproc_include | preproc_def | preproc_function_def | preproc_call)
preproc_include: (/#[ \t]*include/ (STRING_LITERAL | SYSTEM_LIB_STRING | identifier | preproc_call_expression) "\n")
preproc_def: (/#[ \t]*define/ identifier [preproc_arg] "\n")
preproc_function_def: (/#[ \t]*define/ identifier preproc_params [preproc_arg] "\n")
preproc_params: ("(" [((identifier | "...") (("," (identifier | "...")))*)] ")")
preproc_call: (preproc_directive [preproc_arg] "\n")
preproc_if: (/#[ \t]*if/ _preproc_expression "\n" (_top_level_item)* [(preproc_else | preproc_elif)] /#[ \t]*endif/)
preproc_ifdef: ((/#[ \t]*ifdef/ | /#[ \t]*ifndef/) identifier (_top_level_item)* [(preproc_else | preproc_elif)] /#[ \t]*endif/)
preproc_else: (/#[ \t]*else/ (_top_level_item)*)
preproc_elif: (/#[ \t]*elif/ _preproc_expression "\n" (_top_level_item)* [(preproc_else | preproc_elif)])
preproc_if_in_field_declaration_list: (/#[ \t]*if/ _preproc_expression "\n" (_field_declaration_list_item)* [(preproc_else_in_field_declaration_list | preproc_elif_in_field_declaration_list)] /#[ \t]*endif/)
preproc_ifdef_in_field_declaration_list: ((/#[ \t]*ifdef/ | /#[ \t]*ifndef/) identifier (_field_declaration_list_item)* [(preproc_else_in_field_declaration_list | preproc_elif_in_field_declaration_list)] /#[ \t]*endif/)
preproc_else_in_field_declaration_list: (/#[ \t]*else/ (_field_declaration_list_item)*)
preproc_elif_in_field_declaration_list: (/#[ \t]*elif/ _preproc_expression "\n" (_field_declaration_list_item)* [(preproc_else_in_field_declaration_list | preproc_elif_in_field_declaration_list)])
preproc_directive: /#[ \t]*[a-zA-Z]\w*/
preproc_arg: /.+|\\\r?\n/
_preproc_expression: (identifier | preproc_call_expression | number_literal | char_literal | preproc_defined | preproc_unary_expression | preproc_binary_expression | preproc_parenthesized_expression)
preproc_parenthesized_expression: ("(" _preproc_expression ")")
preproc_defined: (("defined" "(" identifier ")") | ("defined" identifier))
preproc_unary_expression: (("!" | "~" | "-" | "+") _preproc_expression)
preproc_call_expression: (identifier preproc_argument_list)
preproc_argument_list: ("(" [(_preproc_expression (("," _preproc_expression))*)] ")")
preproc_binary_expression: ((_preproc_expression "+" _preproc_expression) | (_preproc_expression "-" _preproc_expression) | (_preproc_expression "*" _preproc_expression) | (_preproc_expression "/" _preproc_expression) | (_preproc_expression "%" _preproc_expression) | (_preproc_expression "||" _preproc_expression) | (_preproc_expression "&&" _preproc_expression) | (_preproc_expression "|" _preproc_expression) | (_preproc_expression "^" _preproc_expression) | (_preproc_expression "&" _preproc_expression) | (_preproc_expression "==" _preproc_expression) | (_preproc_expression "!=" _preproc_expression) | (_preproc_expression ">" _preproc_expression) | (_preproc_expression ">=" _preproc_expression) | (_preproc_expression "<=" _preproc_expression) | (_preproc_expression "<" _preproc_expression) | (_preproc_expression "<<" _preproc_expression) | (_preproc_expression ">>" _preproc_expression))
function_definition: ([ms_call_modifier] _declaration_specifiers _declarator compound_statement)
declaration: (_declaration_specifiers ((_declarator | init_declarator) (("," (_declarator | init_declarator)))*) ";")
type_definition: ("typedef" (type_qualifier)* _type_specifier (_type_declarator (("," _type_declarator))*) ";")
_declaration_specifiers: (((storage_class_specifier | type_qualifier | attribute_specifier | ms_declspec_modifier))* _type_specifier ((storage_class_specifier | type_qualifier | attribute_specifier | ms_declspec_modifier))*)
linkage_specification: ("extern" STRING_LITERAL (function_definition | declaration | declaration_list))
attribute_specifier: ("__attribute__" "(" argument_list ")")
ms_declspec_modifier: ("__declspec" "(" identifier ")")
ms_based_modifier: ("__based" argument_list)
ms_call_modifier: ("__cdecl" | "__clrcall" | "__stdcall" | "__fastcall" | "__thiscall" | "__vectorcall")
ms_restrict_modifier: "__restrict"
ms_unsigned_ptr_modifier: "__uptr"
ms_signed_ptr_modifier: "__sptr"
ms_unaligned_ptr_modifier: ("_unaligned" | "__unaligned")
ms_pointer_modifier: (ms_unaligned_ptr_modifier | ms_restrict_modifier | ms_unsigned_ptr_modifier | ms_signed_ptr_modifier)
declaration_list: ("{" (_top_level_item)* "}")
_declarator: (pointer_declarator | function_declarator | array_declarator | parenthesized_declarator | identifier)
_field_declarator: (pointer_field_declarator | function_field_declarator | array_field_declarator | parenthesized_field_declarator | _field_identifier)
_type_declarator: (pointer_type_declarator | function_type_declarator | array_type_declarator | parenthesized_type_declarator | _type_identifier)
_abstract_declarator: (abstract_pointer_declarator | abstract_function_declarator | abstract_array_declarator | abstract_parenthesized_declarator)
parenthesized_declarator: ("(" _declarator ")")
parenthesized_field_declarator: ("(" _field_declarator ")")
parenthesized_type_declarator: ("(" _type_declarator ")")
abstract_parenthesized_declarator: ("(" _abstract_declarator ")")
pointer_declarator: ([ms_based_modifier] "*" (ms_pointer_modifier)* (type_qualifier)* _declarator)
pointer_field_declarator: ([ms_based_modifier] "*" (ms_pointer_modifier)* (type_qualifier)* _field_declarator)
pointer_type_declarator: ([ms_based_modifier] "*" (ms_pointer_modifier)* (type_qualifier)* _type_declarator)
abstract_pointer_declarator: ("*" (type_qualifier)* [_abstract_declarator])
function_declarator: (_declarator parameter_list (attribute_specifier)*)
function_field_declarator: (_field_declarator parameter_list)
function_type_declarator: (_type_declarator parameter_list)
abstract_function_declarator: ([_abstract_declarator] parameter_list)
array_declarator: (_declarator "[" (type_qualifier)* [(_expression | "*")] "]")
array_field_declarator: (_field_declarator "[" (type_qualifier)* [(_expression | "*")] "]")
array_type_declarator: (_type_declarator "[" (type_qualifier)* [(_expression | "*")] "]")
abstract_array_declarator: ([_abstract_declarator] "[" (type_qualifier)* [(_expression | "*")] "]")
init_declarator: (_declarator "=" (initializer_list | _expression))
compound_statement: ("{" (_top_level_item)* "}")
storage_class_specifier: ("extern" | "static" | "auto" | "register" | "inline")
type_qualifier: ("const" | "volatile" | "restrict" | "_Atomic")
_type_specifier: (struct_specifier | union_specifier | enum_specifier | macro_type_specifier | sized_type_specifier | primitive_type | _type_identifier)
sized_type_specifier: ([("signed" | "unsigned" | "long" | "short")] [(_type_identifier | primitive_type)])
primitive_type: ("bool" | "char" | "int" | "float" | "double" | "void" | "size_t" | "ssize_t" | "intptr_t" | "uintptr_t" | "charptr_t" | "int8_t" | "int16_t" | "int32_t" | "int64_t" | "uint8_t" | "uint16_t" | "uint32_t" | "uint64_t" | "char8_t" | "char16_t" | "char32_t" | "char64_t")
enum_specifier: ("enum" ((_type_identifier [enumerator_list]) | enumerator_list))
enumerator_list: ("{" [(enumerator (("," enumerator))*)] [","] "}")
struct_specifier: ("struct" [ms_declspec_modifier] ((_type_identifier [field_declaration_list]) | field_declaration_list))
union_specifier: ("union" [ms_declspec_modifier] ((_type_identifier [field_declaration_list]) | field_declaration_list))
field_declaration_list: ("{" (_field_declaration_list_item)* "}")
_field_declaration_list_item: (field_declaration | preproc_def | preproc_function_def | preproc_call | preproc_if_in_field_declaration_list | preproc_ifdef_in_field_declaration_list)
field_declaration: (_declaration_specifiers [(_field_declarator (("," _field_declarator))*)] [bitfield_clause] ";")
bitfield_clause: (":" _expression)
enumerator: (identifier [("=" _expression)])
parameter_list: ("(" [((parameter_declaration | "...") (("," (parameter_declaration | "...")))*)] ")")
parameter_declaration: (_declaration_specifiers [(_declarator | _abstract_declarator)])
_statement: (case_statement | _non_case_statement)
_non_case_statement: (labeled_statement | compound_statement | expression_statement | if_statement | switch_statement | do_statement | while_statement | for_statement | return_statement | break_statement | continue_statement | goto_statement)
labeled_statement: (_statement_identifier ":" _statement)
expression_statement: ([(_expression | comma_expression)] ";")
if_statement: ("if" parenthesized_expression _statement [("else" _statement)])
switch_statement: ("switch" parenthesized_expression compound_statement)
case_statement: ((("case" _expression) | "default") ":" ((_non_case_statement | declaration | type_definition))*)
while_statement: ("while" parenthesized_expression _statement)
do_statement: ("do" _statement "while" parenthesized_expression ";")
for_statement: ("for" "(" (declaration | ([(_expression | comma_expression)] ";")) [_expression] ";" [(_expression | comma_expression)] ")" _statement)
return_statement: ("return" [(_expression | comma_expression)] ";")
break_statement: ("break" ";")
continue_statement: ("continue" ";")
goto_statement: ("goto" _statement_identifier ";")
_expression: (conditional_expression | assignment_expression | binary_expression | unary_expression | update_expression | cast_expression | pointer_expression | sizeof_expression | subscript_expression | call_expression | field_expression | compound_literal_expression | identifier | number_literal | STRING_LITERAL | true | false | null | concatenated_string | char_literal | parenthesized_expression)
comma_expression: (_expression "," (_expression | comma_expression))
conditional_expression: (_expression "?" _expression ":" _expression)
_assignment_left_expression: (identifier | call_expression | field_expression | pointer_expression | subscript_expression | parenthesized_expression)
assignment_expression: (_assignment_left_expression ("=" | "*=" | "/=" | "%=" | "+=" | "-=" | "<<=" | ">>=" | "&=" | "^=" | "|=") _expression)
pointer_expression: (("*" | "&") _expression)
unary_expression: (("!" | "~" | "-" | "+") _expression)
binary_expression: ((_expression "+" _expression) | (_expression "-" _expression) | (_expression "*" _expression) | (_expression "/" _expression) | (_expression "%" _expression) | (_expression "||" _expression) | (_expression "&&" _expression) | (_expression "|" _expression) | (_expression "^" _expression) | (_expression "&" _expression) | (_expression "==" _expression) | (_expression "!=" _expression) | (_expression ">" _expression) | (_expression ">=" _expression) | (_expression "<=" _expression) | (_expression "<" _expression) | (_expression "<<" _expression) | (_expression ">>" _expression))
update_expression: ((("--" | "++") _expression) | (_expression ("--" | "++")))
cast_expression: ("(" type_descriptor ")" _expression)
type_descriptor: ((type_qualifier)* _type_specifier (type_qualifier)* [_abstract_declarator])
sizeof_expression: ("sizeof" (_expression | ("(" type_descriptor ")")))
subscript_expression: (_expression "[" _expression "]")
call_expression: (_expression argument_list)
argument_list: ("(" [(_expression (("," _expression))*)] ")")
field_expression: ((_expression ("." | "->")) _field_identifier)
compound_literal_expression: ("(" type_descriptor ")" initializer_list)
parenthesized_expression: ("(" (_expression | comma_expression) ")")
initializer_list: ("{" [((initializer_pair | _expression | initializer_list) (("," (initializer_pair | _expression | initializer_list)))*)] [","] "}")
initializer_pair: ([(subscript_designator | field_designator)] "=" (_expression | initializer_list))
subscript_designator: ("[" _expression "]")
field_designator: ("." _field_identifier)

%import common.NUMBER
%import common.SIGNED_NUMBER
number_literal: NUMBER | SIGNED_NUMBER


char_literal: (("L'" | "u'" | "U'" | "u8'" | "'") (ESCAPE_SEQUENCE | /[^\n']/) "'")
concatenated_string: (STRING_LITERAL [STRING_LITERAL])
STRING_LITERAL: (("L\"" | "u\"" | "U\"" | "u8\"" | "\"") ((/[^\"\n]+/ | ESCAPE_SEQUENCE))* "\"")
ESCAPE_SEQUENCE: ("\\" (/[^xuU]/ | /\d{2,3}/ | /x[0-9a-fA-F]{2,}/ | /u[0-9a-fA-F]{4}/ | /U[0-9a-fA-F]{8}/))
SYSTEM_LIB_STRING: ("<" ((/[^>\n]/ | "\\>"))* ">")
true: ("TRUE" | "true")
false: ("FALSE" | "false")
null: "NULL"
identifier: /[a-zA-Z_]\w*/
_type_identifier: identifier
_field_identifier: identifier
_statement_identifier: identifier
_empty_declaration: (_type_specifier ";")
macro_type_specifier: (identifier "(" type_descriptor ")")
COMMENT: (("//" /(\\(.|\r?\n)|[^\\\n])*/) | ("/*" /[^*]*\*+([^\/*][^*]*\*+)*/ "/"))

%import common.WS
%import common.WS_INLINE
%ignore WS
// %ignore WS_INLINE
%ignore COMMENT
"""
# gram = reformat_grammar(lark_grammar)

parser = lark.Lark(
    lark_grammar,
    # parser="lalr",
    parser="earley",
    # parser="cyk",
    # lexer_callbacks={"COMMENT": comments.append},
    # ambiguity='explicit',
    # lexer='standard',
    # lexer="dynamic_complete",
)

print(parser.parse("""
""").pretty())
