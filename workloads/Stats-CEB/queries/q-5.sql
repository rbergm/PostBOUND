SELECT COUNT(*)
FROM comments AS c, votes AS v
WHERE c.UserId = v.UserId
  AND c.Score = 0;
