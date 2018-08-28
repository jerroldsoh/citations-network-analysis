import argparse
import os
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import csv
import datetime as dt
import logging
import re
from nltk.tokenize import RegexpTokenizer
import string
from unidecode import unidecode

# usage: python reported_j_parser.py path/to/dir/with/html/files/ --log_level --text --text_html --full_html

OUTFILE_NAME_PREFIX = os.path.splitext(os.path.basename(__file__))[0]
TIMESTAMP = dt.datetime.now().strftime('%Y.%m.%d_%Hh%Mm%Ss')

class CaseProcessingError(Exception):
    pass

class ReportedCaseProcessor():

    def __init__(self, output_text, output_text_html, output_full_html):
        self.output_text = output_text
        self.output_text_html = output_text_html
        self.output_full_html = output_full_html
        self.file_to_process = None # init at None so can call process_case directly without previous methods

        self.headers_to_output = [
            'file_name',
            'case_title',
            'all_citations',
            'reporter_citation',
            'neutral_citation',
            'decision_date',
            'court',
            'coram',
            'dissenting_judge',
            'raw_parties',
            'counsel',
            'suit_num_verbose',
            'clean_catch_words',
            'ls_hn_facts',
            'ls_hn_holdings',
            'ls_local_cases',
            'ls_forgn_cases',
            'ls_legis', # slr doesnt correctly distinguish between foreign and local legis
            'topleft_tags',
            'judgment_word_count',
            'assoc_decis',
        ]

        if self.output_text:
            self.headers_to_output.append('judgment_paras')
        if self.output_full_html:
            self.headers_to_output.append('raw_html')
        if self.output_text_html:
            self.headers_to_output.append('judgment_paras_html')

    def process(self, directory, outdir):

        self.directory = directory
        self.outdir = outdir

        files_to_process = []
        for f in os.listdir(directory):
            if f.endswith('.html'):
                files_to_process.append(f)

        self.process_files_while_saving(files_to_process)


    def process_files_while_saving(self, files_to_process):

        full_filename = self.outdir + OUTFILE_NAME_PREFIX + '_{}' + TIMESTAMP + '.csv'

        add_on_fields = ''
        if self.output_text:
            add_on_fields += 'text_'
        if self.output_text_html:
            add_on_fields += 'texthtml_'
        if self.output_full_html:
            add_on_fields += 'fullhtml_'
        
        with open(full_filename.format(add_on_fields), 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=self.headers_to_output)
            csv_writer.writeheader()

            num_files_to_process = len(files_to_process)
            for idx, file_to_process in enumerate(files_to_process):

                self.file_to_process = file_to_process

                logging.info('Processing file {} of {}: {}'.format(str(idx+1), num_files_to_process, str(file_to_process)))
                with open(directory + str(file_to_process), 'r', encoding='utf-8') as infile:
                    raw_html = infile.read()

                try:
                    case = self.process_case(raw_html)
                    csv_writer.writerow({fieldname:case[fieldname] for fieldname in self.headers_to_output})

                except CaseProcessingError as e:
                    logging.error('Error processing file: {}'.format(file_to_process))
                    print(e)

    def remove_non_ascii(self, text):
        return "".join([x for x in text if ord(x) < 128])

    def process_case(self, raw_html):

        soup = BeautifulSoup(raw_html, 'html.parser')

        case = {}
        case['file_name'] = self.file_to_process
        
        # ROOM FOR IMPROVEMENT HERE
        # handle detection of reporter / neutral cites. in long run we shud
        # just use a reference dict of courts and reporters
        case['all_citations'] = self.get_all_citations(soup)
        case['reporter_citation'] = None
        case['neutral_citation'] = None

        if len(case['all_citations']) == 2:
            case['reporter_citation'] = case['all_citations'][0]
            if not case['neutral_citation']:
                case['neutral_citation'] = case['all_citations'][1]            
        elif len(case['all_citations']) == 1:
            if 'slr' in case['all_citations'][0].lower():
                case['reporter_citation'] = case['all_citations'][0]
            # assumes (sometimes wrongly) that if there is only one citation
            # then it is a neutral one
            if not case['neutral_citation']:
                case['neutral_citation'] = case['all_citations'][0]
        if not case['reporter_citation']:                 
            case['reporter_citation'] = self.get_reporter_citation(soup)
        if not case['neutral_citation']:                 
            case['neutral_citation'] = self.get_neutral_citation(soup)          
        
        case['case_title'] = self.get_case_title(soup)
        case['decision_date'] = self.get_decision_date(soup)
        case['court'] = self.get_court(soup)
        case['coram'] = self.get_coram(soup)
        case['raw_parties'] = self.get_raw_parties(soup)
        case['dissenting_judge'] =  self.get_dissenting_judge(soup)
        case['counsel'] = self.get_counsel(soup)
        case['suit_num_verbose'] = self.get_suit_num_verbose(soup)
        case['clean_catch_words'] = self.get_clean_catchwords(soup)
        case['ls_hn_facts'] = self.get_ls_text_from_class('p', 'HN-Facts', soup)
        case['ls_hn_holdings'] = self.get_ls_text_from_class('p', 'HN-Held', soup)
        case['ls_local_cases'] = self.get_ls_case_and_treatment(soup, local=True)
        case['ls_forgn_cases'] = self.get_ls_case_and_treatment(soup, local=False)

        # note that lawnet tags the foreign local wrongly, so this ordering provides only a guide
        case['ls_legis'] = (self.get_ls_text_from_class('p', 'Local-LegisRefdTo', soup) 
            + self.get_ls_text_from_class('p', 'Foreign-LegisRefdTo', soup))

        case['topleft_tags'] = self.get_topleft_tags(soup)
        case['assoc_decis'] = self.get_assoc_decis(soup)
        
        case_paras = self.get_judgment_paras(soup)
        case['judgment_word_count'] = self.get_judgment_word_count(case_paras) if case_paras else None
        if self.output_text:
            case['judgment_paras'] = ' '.join(unidecode(para.text) for para in case_paras)
        if self.output_text_html:
            case['judgment_paras_html'] = [unidecode(para.prettify()) for para in case_paras]
        if self.output_full_html:
            case['raw_html'] = raw_html

        return case

    def write_all_to_csv(self, cases):

        full_filename = self.directory + OUTFILE_NAME_PREFIX + '_' + 'with_text_heavy' + '_' + TIMESTAMP + '.csv'
        
        with open(full_filename, 'w', newline='', encoding='utf8') as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=FIELDNAMES_WITH_TEXT_HEAVY_FIELDS)
            csv_writer.writeheader()

            for case in cases:
                csv_writer.writerow(case)

    # general flow of individual getter: try finding using the most specific selector, and continuously fall back, if in the end nothing is found, put sentinel value
    def get_all_citations(self, soup):
        elem = soup.find_all('span', class_='Citation')
        if elem:
            return [cite.text.strip() for cite in elem]
        # if not elem:
        #     elem = soup.find('span', class_='NCitation')
        if not elem:
            elem = soup.find('span', class_='title')
            if elem:
                title_text = elem.next_element.next_element.next_element.text
                citation_string = '[' + '['.join(title_text.split('[')[1:])
                return citation_string.split(';')
        if not elem:
            elem = soup.title # head title as last resort, then remove the case name
            if elem:
                citation_string = '[' + '['.join(elem.text.split('[')[1:])
                return citation_string.split(';')
            # join() provides for if multiple citations occur
        if not elem:
            logging.debug("No citation found for {}".format(self.file_to_process))
            return None

        return None

    # fallback methods to get citations if the all_citations method fails
    # these methods must directly extract the citation
    def get_reporter_citation(self, soup):
        elem = soup.find('span', class_='NCitation')
        if not elem:
            return None
        if 'slr' in elem.text:
            return elem.text
        return None

    def get_neutral_citation(self, soup):
        elem = soup.find('span', class_='NCitation')
        if not elem:
            return None
        return elem.text

    def get_case_title(self, soup):

        elem = soup.find('span', class_='caseTitle')
        if not elem:
            # note that we want BODY title, as the HEAD title has citation also
            elem = soup.find('body').title
        if not elem:
            # assumes first thing appearing after is the case title
            # example: <span class='title'>A v B<br>[2018] SGHC 1</span>
            # we only want A v B
            elem = soup.find('span', class_='title')
            return elem.next_element
        if not elem:
            elem = soup.title # head title as last resort
            return elem.text.split('[')[0].strip()
        if not elem:
            logging.debug("No case_title found for {}".format(self.file_to_process))
            return None

        return elem.text.strip()

    #TO CONSIDER: Refactor all getters for fields like suit_num, raw_parties, counsel, into the same method as they are almost identical
    #TRADEOFF: Makes it more reliant on the lawnet table format, more difficult to incorporate more code for specific fields
    def get_decision_date(self, soup):

        elem = soup.find('td', text='Decision Date')

        if not elem:
            logging.debug("No decision_date found for {}".format(self.file_to_process))
            return None

        decision_date = elem.parent.text.split(':')[-1].strip()

        if decision_date == "—": #a character that sometimes is entered as the decision date
            return None
        
        decision_date = decision_date.replace('Febuary', 'February') # a very common typo in the database
        try:
            decision_date_as_date_object = dt.datetime.strptime(decision_date, '%d %B %Y')
        except ValueError:
            decision_date_as_date_object = dt.datetime.strptime(decision_date, '%d %b %Y') # if shortform month name is used
        decision_date_in_sqlite_date_str_format = decision_date_as_date_object.strftime('%Y-%m-%d')

        return decision_date_in_sqlite_date_str_format

    def get_court(self, soup):

        elem = soup.find('td', text='Tribunal/Court')

        if not elem:
            logging.debug("No court elem found for {}".format(self.file_to_process))
            return None

        return elem.parent.text.split(':')[-1].strip()

    def get_coram(self, soup):

        elem = soup.find('td', text='Coram')

        if not elem:
            logging.debug("No coram elem found for {}".format(self.file_to_process))
            return None

        return elem.parent.text.split(':')[-1].strip()

    def get_dissenting_judge(self, soup):

        paragraphs = soup.find_all(class_=re.compile(r'HN-Heading|Judg-Author'))

        if not paragraphs:
            logging.debug("No paras for dissenting found for {}".format(self.file_to_process))
            return None

        dissenting_paragraphs = []
        for para in paragraphs:
            if re.search(r'dissenting', para.text, flags=re.IGNORECASE):
                dissenting_paragraphs.append(para.text)
        return dissenting_paragraphs

    def get_counsel(self, soup):

        elem = soup.find('td', text='Counsel Name(s)')

        if not elem:
            logging.debug("No counsel elem found for {}".format(self.file_to_process))
            return None

        counsel = elem.parent.text.split(':')[-1].strip()

        if counsel == "—": #a character that sometimes is entered as the decision date
            return None

        return counsel

    def get_raw_parties(self, soup):
        elem = soup.find('td', text='Parties')

        if not elem:
            logging.debug("No raw_parties elem found for {}".format(self.file_to_process))
            return None

        raw_parties = elem.parent.text.split(':')[-1].strip()

        if raw_parties == "—": #a character that sometimes is entered as the decision date
            return None

        return unidecode(raw_parties)

    def get_suit_num_verbose(self, soup):

        elem = soup.find("td", text="Case Number")

        if not elem:
            logging.debug("No suit_num_verbose elem found for {}".format(self.file_to_process))
            return None

        suit_num_verbose = elem.parent.text.split(":")[1].strip()

        if suit_num_verbose == "—": #a character that sometimes is entered as the decision date
            return None

        return suit_num_verbose

    #getting the hifenated keywords. seems a little too easy that we can get using only this line. need to validate
    def get_clean_catchwords(self, soup):
        catchwords_html = soup.find_all("p", class_="txt-body")

        #split to get phrases, then strip whitespace at boundaries. This takes a while, probably can optimise
        if not catchwords_html:
            logging.debug("No catchwords elems found for {}".format(self.file_to_process))
            return []

        #\u2013 is unicode dash - split before unidecode because only then can split all
        catchwords = []
        for line in catchwords_html:
            line_text = line.text
            sublines = [unidecode(subline).strip() for subline in line_text.split(u'\u2013')]
            catchwords.append(sublines)

        return catchwords

    # general workhorse for simple get all tag with class_ methods
    def get_ls_text_from_class(self, tag_, class_, soup):
        elems = soup.find_all(tag_, class_=class_)
        if not elems: 
            return []
        return [unidecode(elem.text).strip() for elem in elems]

    def get_ls_case_and_treatment(self, soup, local=True):
        class_ = 'Local-CasesRefdTo' if local else 'Foreign-CasesRefdTo'
        cases = soup.find_all('p', class_=class_)
        ls_case_and_treatment = []
        for case in cases:
            case_title = case.find('i')
            case_cites = ';'.join([unidecode(cite.text) for cite in case.find_all('a')])
            case_treatment = case.find('span', class_='Annotation')

            if not case_cites:
                # sometimes lawnet will have the case citations in top level instead of an anchor
                # so we catch all toplevel stuff that are strings and not tags
                case_cites = ''.join([x for x in case.contents if isinstance(x, NavigableString)])
                # sometimes the brackets that are supposed to surround the case treatment 
                # get bumped to toplevel also
                case_cites = case_cites.rstrip('()').strip()

            # each case becomes a triple
            ls_case_and_treatment.append((unidecode(case_title.text),
                unidecode(case_cites),
                unidecode(case_treatment.text)))
        return ls_case_and_treatment

    def get_topleft_tags(self, soup):
        #to preserve hierarchy, need to approach step by step at the right level
        #each top level tag is an li in the container ul of class .filetree, so we find all its siblings
        tree_container = soup.find(class_="filetree")
        if not tree_container:
            logging.debug("No topleft_tags container found for {}".format(self.file_to_process))
            return []
        li_first = tree_container.find("li" )
        tags = []
        tags.append(li_first.text.split("  ")) #.text returns THREE spaces between each tag
        li_rest = li_first.find_next_siblings("li")
        for li in li_rest:
            tags.append(li.text.split("  "))

        #clean away whitespaces - need to go one level in
        tags = [[tag.strip() for tag in ls_tag] for ls_tag in tags]

        #delete blanks that arise due to splitting on "  "
        for i in range(0, len(tags)):
            tags[i] = [word for word in tags[i] if word]

        if not tags:
            logging.debug("No topleft_tags found for {}".format(self.file_to_process))
            return []

        return tags

    def get_judgment_paras(self, soup):
        #gets ONLY WHAT THE JUDGE WROTE, INCLUDING PARA NUMBERS AND QUOTES, SCHEDULES 
        #EXCLUDING FOOTNOTES and lawnet top formatting, HEADINGS IN SCHEDULES

        #identification strategy: LawNet cases may have headnotes, formatting etc, so the only reliable way
        #to get only what the judge writes is to find all <p class="Judg-[]"> where [] includes things like
        #Judg-1, Judg-2, Judg-3
        #Judg-Heading-1, Judg-Heading-2, Judg-Heading-3
        #Judg-Quote-1, Judg-Quote-2
        #Judg-Heading-Quote1
        #A full list of unique classes found within Simon Chesterman's 435 cases is found in 'Chesterman\judg_classes_found_from_435_lawnet_cases.txt'

        judgment_paras = soup.find_all('p', class_=re.compile('Judg-.*\d'))

        if not judgment_paras:
            logging.debug("No judgment_paras found for {}".format(self.file_to_process))
            return []

        #start with a more general regex to see all the possible classes. Eventually we want to move to something more specific like:
        #judgment_paras = soup.find_all('p', class_=re.compile('Judg-(Heading-)?(Quote-)?\d')) 
        #note find_all uses re.search(), which matches from anywhere in string

        # need to remove all imgs else it will screw up the processing
        for para in judgment_paras:
            imgs = para.find_all('img')
            for img in imgs:
                img.decompose() 

        return [unidecode(para) for para in judgment_paras] # keep in list form to preserve para structure

    def get_judgment_word_count(self, judgment_paras):
        judgment_text = ' '.join([para.text for para in judgment_paras])
        tokenizer = RegexpTokenizer(r'\w+') #gives a true word count ignoring punctuation. 
        #Note this will differ slightly from Microsoft Word's wordcount, but it comes pretty close

        cleaned_words = tokenizer.tokenize(judgment_text)
        return len(cleaned_words)

    def get_assoc_decis(self, soup):
        elem = soup.find('span', class_='AssociatedDecision')
        if not elem:
            return None
        return elem.text

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='path to directory with the HTML files from LawNet')
    parser.add_argument('outdir', help='path to directory to dump processed data')
    parser.add_argument('-L', '--log_level', default='INFO')
    parser.add_argument('--text', action='store_true', help='enable to output the judgment text')
    parser.add_argument('--full_html', action='store_true', help='enable to output the full raw html')
    parser.add_argument('--text_html', action='store_true', help='enable to output the judgment text with html tags')
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level, logging.INFO)
    now = dt.datetime.today()
    logging.basicConfig(level=logging.DEBUG)

    # also log to console, setting log level according to commandline input, defaults to logging.INFO
    console = logging.StreamHandler()
    console.setLevel(log_level)
    logging.getLogger('').addHandler(console)

    directory = args.directory
    if not directory[:-1] == '/':
        directory = directory + '/'

    rcp = ReportedCaseProcessor(args.text, args.text_html, args.full_html)
    rcp.process(directory, args.outdir)