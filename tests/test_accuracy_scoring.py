import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from accuracy_scoring import grade
class ScoringTests(unittest.TestCase):
    def case(self,kind,expected,category='format'):return {'grading':kind,'expected':expected,'category':category}
    def test_number_requires_entire_answer_and_finite_value(self):
        c=self.case('number',71,'arithmetic')
        self.assertTrue(grade(c,' 71.0 ')['passed'])
        for bad in ('The answer is 71','17','NaN','inf','71 and 72'):self.assertFalse(grade(c,bad)['passed'])
    def test_json_key_order_is_irrelevant_but_types_and_values_matter(self):
        c=self.case('json',{'count':3,'ok':True})
        self.assertTrue(grade(c,'{"ok":true,"count":3}')['passed'])
        for bad in ('{"count":3,"ok":1}','{"count":4,"ok":true}','{"count":3,"ok":true,"extra":0}'):
            self.assertFalse(grade(c,bad)['passed'])
    def test_fenced_json_preserves_semantic_result_and_fails_format_task(self):
        r=grade(self.case('json',[True,False]),'```json\n[true,false]\n```')
        self.assertTrue(r['semantic_correct']);self.assertFalse(r['format_ok']);self.assertFalse(r['passed'])
    def test_csv_checks_all_items_and_order(self):
        c=self.case('csv_numbers',[2,5,9]);self.assertTrue(grade(c,'2, 5,9')['passed'])
        for bad in ('2,5','9,5,2','2,5,9,10','2,5,9 apples'):self.assertFalse(grade(c,bad)['passed'])
    def test_format_task_preserves_case_requirement(self):
        c=self.case('exact_text','SPARKLE');self.assertTrue(grade(c,'SPARKLE')['passed']);self.assertFalse(grade(c,'sparkle')['passed'])
    def test_text_normalization_does_not_accept_explanations(self):
        c=self.case('text','Paris','factual');self.assertTrue(grade(c,'Paris.')['passed']);self.assertFalse(grade(c,'Paris is the capital')['passed'])
if __name__=='__main__':unittest.main()
