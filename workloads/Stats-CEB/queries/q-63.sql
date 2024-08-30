SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  votes AS v,
  badges AS b,
  users AS u
WHERE p.Id = c.PostId
  AND p.Id = pl.RelatedPostId
  AND p.Id = v.PostId
  AND u.Id = p.OwnerUserId
  AND u.Id = b.UserId
  AND c.Score = 0
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 4
  AND p.CreationDate <= CAST('2014-09-12 15:56:19' AS timestamp)
  AND pl.LinkTypeId = 1
  AND pl.CreationDate >= CAST('2011-03-07 16:05:24' AS timestamp)
  AND v.BountyAmount <= 100
  AND v.CreationDate >= CAST('2009-02-03 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-11 00:00:00' AS timestamp)
  AND u.Views <= 160
  AND u.CreationDate >= CAST('2010-07-27 12:58:30' AS timestamp)
  AND u.CreationDate <= CAST('2014-07-12 20:08:07' AS timestamp);
